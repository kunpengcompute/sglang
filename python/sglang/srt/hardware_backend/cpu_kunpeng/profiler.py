# Copyright 2026 Huawei Technologies Co., Ltd.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import time
import functools
import logging
from typing import Callable, Any, Optional
from sglang.srt.environ import envs

logger = logging.getLogger(__name__)


class KunpengProfiler:
    enabled = False

    def __init__(self, func: Optional[Callable] = None, depth: int = 0) -> None:
        self.func = func
        self.depth = depth

        if func is not None:
            functools.update_wrapper(self, func)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if self.func is None:
            if len(args) == 1 and callable(args[0]) and not kwargs:
                real_func = args[0]
                return KunpengProfiler(real_func, depth=self.depth)
            else:
                raise TypeError("Invalid usage.")

        if not self.enabled:
            return self.func(*args, **kwargs)

        start_time = time.perf_counter()
        result = None
        success = False

        try:
            result = self.func(*args, **kwargs)
            success = True
            return result
        finally:
            if self.enabled:
                end_time = time.perf_counter()
                elapsed_time = end_time - start_time
                func_name = getattr(self.func, "__name__", "unknown")

                if args and hasattr(args[0], "__class__"):
                    if hasattr(type(args[0]), func_name) and callable(
                        getattr(type(args[0]), func_name)
                    ):
                        class_name = args[0].__class__.__name__
                        display_name = f"{class_name}.{func_name}"
                    else:
                        display_name = func_name
                else:
                    display_name = func_name

                indent_str = (
                    "" if self.depth == 0 else ("|   " * (self.depth - 1) + "|---")
                )
                status = "completed" if success else "failed with exception"
                logger.info(
                    f"{indent_str}{display_name} {status} in {elapsed_time * 1000:.3f} ms"
                )

    def __get__(self, instance, owner):
        if instance is None:
            return self

        bound_method = functools.partial(self.__call__, instance)
        functools.update_wrapper(bound_method, self.func)
        return bound_method


if envs.SGLANG_KUNPENG_PROFILE.get():
    KunpengProfiler.enabled = True
