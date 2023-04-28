import sys
import logging

for device in tf.config.experimental.list_physical_devices("GPU"):
    try:
        tf.config.experimental.set_memory_growth(device, True)
    except RuntimeError as e:
        logging.warn(f"can not set {device} to dynamically allocate memory. {e}")
from tensorflow.python.framework.dtypes import DType
from tensorflow.python.framework.tensor_shape import TensorShape
from tensorflow.python.types.core import Tensor
import ivy
from ivy.func_wrapper import _dtype_from_version

backend_version = {"version": tf.__version__}
if not ivy.is_local():
    _module_in_memory = sys.modules[__name__]
else:
    _module_in_memory = sys.modules[ivy.import_module_path].import_cache[__name__]
use = ivy.utils.backend.ContextManager(_module_in_memory)
NativeArray = None
NativeDevice = str
NativeDtype = None
NativeShape = None
NativeSparseArray = None
valid_devices = (["cpu", "gpu"],)
invalid_devices = (["tpu"],)
native_int8 = None
native_int16 = None
native_int32 = None
native_int64 = None
native_uint8 = None
native_uint16 = None
native_uint32 = None
native_uint64 = None
native_bfloat16 = None
native_float16 = None
native_float32 = None
native_float64 = None
native_complex64 = None
native_complex128 = None
native_double = native_float64
native_bool = None
valid_dtypes_dict = {
    "None": (
        "ivy.int8",
        "ivy.int16",
        "ivy.int32",
        "ivy.int64",
        "ivy.uint8",
        "ivy.bfloat16",
        "ivy.float16",
        "ivy.float32",
        "ivy.float64",
        "ivy.complex64",
        "ivy.complex128",
        "ivy.bool",
    )
}
valid_dtypes = _dtype_from_version(valid_dtypes_dict, backend_version)
valid_numeric_dtypes_dict = {
    "None": (
        "ivy.int8",
        "ivy.int16",
        "ivy.int32",
        "ivy.int64",
        "ivy.uint8",
        "ivy.bfloat16",
        "ivy.float16",
        "ivy.float32",
        "ivy.float64",
        "ivy.complex64",
        "ivy.complex128",
    )
}
valid_numeric_dtypes = _dtype_from_version(valid_numeric_dtypes_dict, backend_version)
valid_int_dtypes_dict = {
    "None": ("ivy.int8", "ivy.int16", "ivy.int32", "ivy.int64", "ivy.uint8")
}
valid_int_dtypes = _dtype_from_version(valid_int_dtypes_dict, backend_version)
valid_float_dtypes_dict = {
    "None": ("ivy.bfloat16", "ivy.float16", "ivy.float32", "ivy.float64")
}
valid_float_dtypes = _dtype_from_version(valid_float_dtypes_dict, backend_version)
valid_uint_dtypes_dict = {"None": ("ivy.uint8",)}
valid_uint_dtypes = _dtype_from_version(valid_uint_dtypes_dict, backend_version)
valid_complex_dtypes_dict = {"None": ("ivy.complex64", "ivy.complex128")}
valid_complex_dtypes = _dtype_from_version(valid_complex_dtypes_dict, backend_version)
invalid_dtypes_dict = {"None": ("ivy.uint32", "ivy.uint64", "ivy.uint16")}
invalid_dtypes = _dtype_from_version(invalid_dtypes_dict, backend_version)
invalid_numeric_dtypes_dict = {"None": ("ivy.uint32", "ivy.uint64", "ivy.uint16")}
invalid_numeric_dtypes = _dtype_from_version(
    invalid_numeric_dtypes_dict, backend_version
)
invalid_int_dtypes_dict = {"None": ("ivy.uint16", "ivy.uint32", "ivy.uint64")}
invalid_int_dtypes = _dtype_from_version(invalid_int_dtypes_dict, backend_version)
invalid_float_dtypes_dict = {"None": ()}
invalid_float_dtypes = _dtype_from_version(invalid_float_dtypes_dict, backend_version)
invalid_uint_dtypes_dict = {"None": ("ivy.uint16", "ivy.uint32", "ivy.uint64")}
invalid_uint_dtypes = _dtype_from_version(invalid_uint_dtypes_dict, backend_version)
invalid_complex_dtypes_dict = {"None": ()}
invalid_complex_dtypes = _dtype_from_version(
    invalid_complex_dtypes_dict, backend_version
)
native_inplace_support = False
supports_gradients = True


def closest_valid_dtype(type=None, /, as_native=False):
    raise NotImplementedError("mxnet.closest_valid_dtype Not Implemented")


backend = "mxnet"
from . import activations
from .activations import *
from . import creation
from .creation import *
from . import data_type
from .data_type import *
from . import device
from .device import *
from . import elementwise
from .elementwise import *
from . import general
from .general import *
from . import gradients
from .gradients import *
from . import layers
from .layers import *
from . import linear_algebra as linalg
from .linear_algebra import *
from . import manipulation
from .manipulation import *
from . import random
from .random import *
from . import searching
from .searching import *
from . import set
from .set import *
from . import sorting
from .sorting import *
from . import statistical
from .statistical import *
from . import utility
from .utility import *
from . import experimental
from .experimental import *
from . import control_flow_ops
from .control_flow_ops import *
from . import sub_backends
from .sub_backends import *
