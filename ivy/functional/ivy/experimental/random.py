# local
from typing import Optional, Union, Sequence
import ivy
from ivy.func_wrapper import (
    handle_out_argument,
    to_native_arrays_and_back,
    handle_nestable,
    infer_dtype,
    infer_device,
)
from ivy.exceptions import handle_exceptions


# dirichlet
@to_native_arrays_and_back
@handle_out_argument
@handle_nestable
@handle_exceptions
def dirichlet(
    alpha: Union[ivy.Array, ivy.NativeArray, float, Sequence[float]],
    /,
    *,
    size: Optional[Union[ivy.Shape, ivy.NativeShape]] = None,
    dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
    seed: Optional[int] = None,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """Draw size samples of dimension k from a Dirichlet distribution.
    A Dirichlet-distributed random variable can be seen as a multivariate
    generalization of a Beta distribution. The Dirichlet distribution is
    a conjugate prior of a multinomial distribution in Bayesian inference.

    Parameters
    ----------
    alpha
        Sequence of floats of length k
    size
        optional int or tuple of ints, Output shape. If the given shape is,
        e.g., (m, n), then m * n * k samples are drawn. Default is None,
        in which case a vector of length k is returned.
    dtype
        output array data type. If ``dtype`` is ``None``, the output array data
        type will be the default floating-point data type. Default ``None``
    seed
        A python integer. Used to create a random seed distribution
    out
        optional output array, for writing the result to.

    Returns
    -------
    ret
        The drawn samples, of shape (size, k).

    Functional Examples
    -------------------

    >>> alpha = [1.0, 2.0, 3.0]
    >>> ivy.dirichlet(alpha)
    ivy.array([0.10598304, 0.21537054, 0.67864642])

    >>> alpha = [1.0, 2.0, 3.0]
    >>> ivy.dirichlet(alpha, size = (2,3))
    ivy.array([[[0.48006698, 0.07472073, 0.44521229],
        [0.55479872, 0.05426367, 0.39093761],
        [0.19531053, 0.51675832, 0.28793114]],

       [[0.12315625, 0.29823365, 0.5786101 ],
        [0.15564976, 0.50542368, 0.33892656],
        [0.1325352 , 0.44439589, 0.42306891]]])
    """
    return ivy.current_backend().dirichlet(
        alpha,
        size=size,
        dtype=dtype,
        seed=seed,
        out=out,
    )


@to_native_arrays_and_back
@handle_out_argument
@handle_nestable
@handle_exceptions
def beta(
    a: Union[float, ivy.NativeArray, ivy.Array],
    b: Union[float, ivy.NativeArray, ivy.Array],
    /,
    *,
    shape: Optional[Union[ivy.Shape, ivy.NativeShape]] = None,
    device: Optional[Union[ivy.Device, ivy.NativeDevice]] = None,
    dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
    seed: Optional[int] = None,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """Returns an array filled with random values sampled from a beta distribution.

    Parameters
    ----------
    a
        Alpha parameter of the beta distribution.
    b
        Beta parameter of the beta distribution.
    shape
        If the given shape is, e.g ``(m, n, k)``, then ``m * n * k`` samples are drawn
        Can only be specified when ``mean`` and ``std`` are numeric values, else
        exception will be raised.
        Default is ``None``, where a single value is returned.
    device
        device on which to create the array. 'cuda:0',
        'cuda:1', 'cpu' etc. (Default value = None).
    dtype
        output array data type. If ``dtype`` is ``None``, the output array data
        type will be the default floating point data type. Default ``None``
    seed
        A python integer. Used to create a random seed distribution
    out
        optional output array, for writing the result to. It must have a shape
        that the inputs broadcast to.

    Returns
    -------
    ret
        Returns an array with the given shape filled with random values sampled from
        a beta distribution.
    """
    return ivy.current_backend().beta(
        a, b, shape=shape, device=device, dtype=dtype, seed=seed, out=out
    )


@to_native_arrays_and_back
@handle_out_argument
@handle_nestable
@handle_exceptions
def gamma(
    alpha: Union[float, ivy.NativeArray, ivy.Array],
    beta: Union[float, ivy.NativeArray, ivy.Array],
    /,
    *,
    shape: Union[float, ivy.NativeArray, ivy.Array],
    device: Optional[Union[ivy.Device, ivy.NativeDevice]] = None,
    dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
    seed: Optional[int] = None,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """Returns an array filled with random values sampled from a gamma distribution.

    Parameters
    ----------
    shape
        Shape parameter of the gamma distribution.
    alpha
        Alpha parameter of the gamma distribution.
    beta
        Beta parameter of the gamma distribution.
    device
        device on which to create the array. 'cuda:0',
        'cuda:1', 'cpu' etc. (Default value = None).
    dtype
        output array data type. If ``dtype`` is ``None``, the output array data
        type will be the default floating point data type. Default ``None``
    seed
        A python integer. Used to create a random seed distribution
    out
        optional output array, for writing the result to. It must have a shape
        that the inputs broadcast to.

    Returns
    -------
    ret
        Returns an array filled with random values sampled from a gamma distribution.
    """
    return ivy.current_backend().gamma(
        shape, alpha, beta, device=device, dtype=dtype, seed=seed, out=out
    )


@to_native_arrays_and_back
@handle_out_argument
@infer_device
@infer_dtype
@handle_nestable
@handle_exceptions
def poisson(
    lam: Union[float, ivy.Array, ivy.NativeArray],
    *,
    shape: Optional[Union[ivy.Shape, ivy.NativeShape]] = None,
    device: Optional[Union[ivy.Device, ivy.NativeDevice]] = None,
    dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
    seed: Optional[int] = None,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """
    Draws samples from a poisson distribution.

    Parameters
    ----------
    lam
        Rate parameter(s) describing the poisson distribution(s) to sample.
        It must have a shape that is broadcastable to the requested shape.
    shape
        If the given shape is, e.g '(m, n, k)', then 'm * n * k' samples are drawn.
        (Default value = 'None', where 'ivy.shape(lam)' samples are drawn)
    device
        device on which to create the array 'cuda:0', 'cuda:1', 'cpu' etc.
        (Default value = None).
    dtype
        output array data type. If ``dtype`` is ``None``, the output array data
        type will be the default floating-point data type. Default ``None``
    seed
        A python integer. Used to create a random seed distribution
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
     ret
        Drawn samples from the poisson distribution

    Functional Examples
    -------------------

    >>> lam = [1.0, 2.0, 3.0]
    >>> ivy.poisson(lam)
    ivy.array([1., 4., 4.])

    >>> lam = [1.0, 2.0, 3.0]
    >>> ivy.poisson(lam, shape = (2,3))
    ivy.array([[0., 2., 2.],
               [1., 2., 3.]])
    """
    return ivy.current_backend().poisson(
        lam,
        shape=shape,
        device=device,
        dtype=dtype,
        seed=seed,
        out=out,
    )
