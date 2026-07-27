
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

from sgl_kernel._kupl_async import PyKuplExecutor
from sgl_kernel._kupl_async import PyKuplEgroup

__all__ = ["KuplExecutor", "KuplEgroup"]


class KuplEgroup:
    def __init__(self, executors: list[int]):
        self._impl = PyKuplEgroup(executors)


class KuplExecutor:
    def __init__(self):
        self._impl = PyKuplExecutor()

    def submit(self, fn, *args, egroup=None, **kwargs):
        eg = egroup._impl if egroup is not None else None
        if kwargs:
            self._impl.submit(lambda: fn(*args, **kwargs), egroup=eg)
        else:
            self._impl.submit(fn, *args, egroup=eg)

    def wait(self):
        return self._impl.wait()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False
