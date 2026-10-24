// Generic boxed ATen adapter for the existing registered graph executor.
// First establish coverage/correctness; functional ops may allocate an ATen
// temporary before copying into the graph-planned output.
#include <ATen/ATen.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <atomic>
#include <string>
#include <variant>
#include <vector>
using ScalarArg = std::variant<bool, int64_t, double, std::string>;
static std::atomic<uint64_t> calls{0};

static c10::IValue decode(const std::vector<ScalarArg> &s, size_t &pos, std::vector<at::Tensor> &t,
                          c10::TypePtr expected) {
    auto tag = std::get<std::string>(s.at(pos++));
    // Python accepts None for a required Tensor as an undefined Tensor (the
    // native dense bias schema uses this convention). Boxed replay must keep
    // that distinction from an Optional[Tensor] carrying IValue None.
    if (tag == "N")
        return expected->isSubtypeOf(c10::TensorType::get()) ? c10::IValue(at::Tensor())
                                                             : c10::IValue();
    if (tag == "T")
        return t.at(std::get<int64_t>(s.at(pos++)));
    if (tag == "I")
        return std::get<int64_t>(s.at(pos++));
    if (tag == "F")
        return std::get<double>(s.at(pos++));
    if (tag == "B")
        return std::get<bool>(s.at(pos++));
    if (tag == "S")
        return std::get<std::string>(s.at(pos++));
    if (tag == "D")
        return c10::Device(std::get<std::string>(s.at(pos++)));
    if (auto optional = expected->cast<c10::OptionalType>())
        expected = optional->getElementType();
    if (tag == "L") {
        auto type = expected->cast<c10::ListType>();
        TORCH_CHECK(type, "Graph boxed list has non-list schema");
        auto n = std::get<int64_t>(s.at(pos++));
        c10::impl::GenericList out(type->getElementType());
        for (int64_t i = 0; i < n; ++i)
            out.push_back(decode(s, pos, t, type->getElementType()));
        return out;
    }
    TORCH_CHECK(false, "Unsupported graph argument tag ", tag);
}

static void flatten(const c10::IValue &value, std::vector<at::Tensor> &output) {
    if (value.isTensor())
        output.push_back(value.toTensor());
    else if (value.isTuple())
        for (const auto &v : value.toTupleRef().elements())
            flatten(v, output);
    else if (value.isTensorList())
        for (const auto &v : value.toTensorVector())
            output.push_back(v);
    else
        TORCH_CHECK(false, "Graph boxed result must contain only tensors");
}

extern "C" void registered_aten_dispatch(std::vector<at::Tensor> &tensors,
                                         const std::vector<ScalarArg> &scalars) {
    auto name = std::get<std::string>(scalars.at(0));
    auto overload = std::get<std::string>(scalars.at(1));
    auto input_count = std::get<int64_t>(scalars.at(2));
    auto output_count = std::get<int64_t>(scalars.at(3));
    auto inplace = std::get<bool>(scalars.at(4));
    TORCH_CHECK(name.rfind("aten::", 0) == 0 || name == "sgl_kernel::bf16_linear_kunpeng" ||
                    name == "sgl_kernel::silu_and_mul_cpu" ||
                    name == "sgl_kernel::fused_qkvzba_split_reshape_cat_contiguous_cpu",
                "Boxed graph operator has no approved native shape contract");
    auto op = c10::Dispatcher::singleton().findSchemaOrThrow(name.c_str(), overload.c_str());
    std::vector<c10::IValue> stack;
    size_t pos = 5;
    for (const auto &argument : op.schema().arguments())
        stack.push_back(decode(scalars, pos, tensors, argument.type()));
    TORCH_CHECK(pos == scalars.size(), "Trailing graph argument descriptors");
    op.callBoxed(&stack);
    if (inplace) {
        TORCH_CHECK(output_count == 0 && stack.size() == 1 && stack[0].isTensor() &&
                        stack[0].toTensor().data_ptr() == tensors.at(0).data_ptr(),
                    "Unexpected graph in-place alias result");
    } else {
        std::vector<at::Tensor> results;
        for (const auto &value : stack)
            flatten(value, results);
        TORCH_CHECK(static_cast<int64_t>(results.size()) == output_count &&
                        tensors.size() == static_cast<size_t>(input_count + output_count),
                    "Graph boxed outputs mismatch");
        for (int64_t i = 0; i < output_count; ++i) {
            auto &destination = tensors.at(input_count + i);
            TORCH_CHECK(results[i].sizes() == destination.sizes() &&
                            results[i].scalar_type() == destination.scalar_type(),
                        "Graph boxed output contract changed");
            destination.copy_(results[i]);
        }
    }
    ++calls;
}
extern "C" uint64_t registered_aten_calls() { return calls.load(); }
