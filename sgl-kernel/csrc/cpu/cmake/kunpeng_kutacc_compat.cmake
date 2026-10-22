# Recent SGLang optional kernels may precede their APIs in public KuTACC.
# Strict upstream compilation remains the default. Opting in retains the native
# implementation when its API exists, otherwise the adapter raises on use.
option(SGLANG_KUNPENG_ALLOW_MISSING_OPTIONAL_KUTACC_APIS
    "Build unsupported optional KuTACC adapters as explicit runtime errors" OFF)
if(SGLANG_KUNPENG_ALLOW_MISSING_OPTIONAL_KUTACC_APIS)
    include(CheckCXXSourceCompiles)
    set(_saved_try_compile_target_type "${CMAKE_TRY_COMPILE_TARGET_TYPE}")
    set(_saved_required_includes "${CMAKE_REQUIRED_INCLUDES}")
    set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)
    set(CMAKE_REQUIRED_INCLUDES "${KUTACC_INCLUDE}")
    function(kunpeng_optional_kutacc_api feature source body)
        # A new library/header can be installed into the same prefix.
        unset(${feature} CACHE)
        check_cxx_source_compiles(
            "#include <array>\n#include <tuple>\n#include <kutacc.h>\nint main() { ${body} return 0; }"
            ${feature})
        if(${feature})
            set(value 1)
        else()
            set(value 0)
            message(WARNING "${feature}: unavailable; ${source} will raise if this optional API is requested")
        endif()
        set_property(SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/${source}"
            APPEND PROPERTY COMPILE_DEFINITIONS "${feature}=${value}")
    endfunction()
    kunpeng_optional_kutacc_api(SGLANG_KUTACC_HAS_DSA_INDEXER
        cpu_kunpeng/attention/indexer.cpp
        "kutacc::Tensor<bfloat16_t,3> q; kutacc::Tensor<bfloat16_t,2> w;
         kutacc::Tensor<int,2> table, out; kutacc::Tensor<int,1> lens;
         kutacc::Tensor<uint8_t,1> workspace; kutacc::attention_schedule s;
         s.mode=kutacc::attention_schedule::Mode::kSplitKV;
         s.output_coord=kutacc::attention_schedule::OutputCoord::kPosition;
         s.row_start=0; s.row_rows=-1;
         kutacc::paged_mqa_logits_topk(q,q,w,table,lens,lens,out,s,workspace,1);")
    kunpeng_optional_kutacc_api(SGLANG_KUTACC_HAS_FULL_ALLGATHER
        cpu_kunpeng/comm/allgather.cpp
        "kutacc::kurmcl_conn_info_h c=nullptr;
         kutacc::kurmcl_allgather_full_init(nullptr,0,nullptr,0,c);
         kutacc::kurmcl_allgather_full(nullptr,0,nullptr,0,c);
         kutacc::kurmcl_allgather_full_finalize();")
    kunpeng_optional_kutacc_api(SGLANG_KUTACC_HAS_BROADCAST
        cpu_kunpeng/comm/broadcast.cpp
        "kutacc::kurmcl_conn_info_h c=nullptr; kutacc::kurmcl_broadcast(nullptr,0,0,0,c);")
    kunpeng_optional_kutacc_api(SGLANG_KUTACC_HAS_MLA_LONG_CONTEXT_READ
        cpu_kunpeng/comm/alltoall_long_context.cpp
        "std::vector<uint8_t*> peers; kutacc::mla_o_alltoall_long_context_read_kernel(
            static_cast<bfloat16_t*>(nullptr),static_cast<float*>(nullptr),0,8,512,peers,0,8,0);")
    kunpeng_optional_kutacc_api(SGLANG_KUTACC_HAS_MLA_REDUCE
        cpu_kunpeng/comm/flash_mla_reduce.cpp
        "kutacc::Tensor<bfloat16_t,3> x; kutacc::Tensor<float,2> lse;
         kutacc::Tensor<bfloat16_t,2> out; kutacc::flash_mla_reduce(x,lse,out);")
    kunpeng_optional_kutacc_api(SGLANG_KUTACC_HAS_BATCHED_BF16_GEMM
        cpu_kunpeng/matmul/bgemm.cpp
        "kutacc::batch_bf16_bf16_packed_gemm_bf16(1,1,32,32,32,32,
            static_cast<bfloat16_t*>(nullptr),static_cast<bfloat16_t*>(nullptr),
            static_cast<bfloat16_t*>(nullptr));")
    kunpeng_optional_kutacc_api(SGLANG_KUTACC_HAS_CTRL_DOMAIN
        cpu_kunpeng/moe/kunpeng_moe.cpp
        "kutacc::kurmcl_conn_info_h c=nullptr;
         kutacc::kurmcl_comm_create_independent(1,0,nullptr,nullptr,&c);")
    kunpeng_optional_kutacc_api(SGLANG_KUTACC_HAS_MOE_COMBINE_REG_SIZE
        cpu_kunpeng/moe/kunpeng_moe.cpp
        "std::vector<bfloat16_t*> send, recv;
         kutacc::moe_combine_init(nullptr,0,0,0,0,0,std::move(send),0,
                                 std::move(recv),nullptr,0);")
    set(CMAKE_REQUIRED_INCLUDES "${_saved_required_includes}")
    if(_saved_try_compile_target_type)
        set(CMAKE_TRY_COMPILE_TARGET_TYPE "${_saved_try_compile_target_type}")
    else()
        unset(CMAKE_TRY_COMPILE_TARGET_TYPE)
    endif()
endif()
