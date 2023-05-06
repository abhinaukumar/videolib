from typing import Tuple, Union
import numpy as np

from . import nonlinearities
from . import utils


def cat02(xyz: np.ndarray, xyz_white: np.ndarray, D: float, FL: float) -> Tuple[np.ndarray, np.ndarray]:
    '''
    CAT02 Color Adaption Transform

    Args:
        xyz: Tristrimulus XYZ image to adapt.
        xyz_white: White point to adapt input to.
        D: Degree of adaptation.
        FL: Luminance level adaptation factor.
    Returns:
        Adapted LMS responses for image and white point.
    '''
    transfer_mat_cat02 = np.array([[0.7328, 0.4296, -0.1624], [-0.7036, 1.6975, 0.0061], [0.0030, 0.0136, 0.9834]])
    lms_white = transfer_mat_cat02 @ xyz_white
    # Updating transfer matrix to adapt to white point
    for i in range(3):
        transfer_mat_cat02[i, :] = (xyz_white[1]/lms_white[i] * D + 1 - D) * transfer_mat_cat02[i, :]

    lms_c = utils.apply_transfer_mat(xyz, transfer_mat_cat02)

    # Convert to the Hunt-Pointer-Estevez space
    transfer_mat_h = np.array([[0.38971, 0.68898, -0.07868], [-0.22981, 1.18340, 0.04641], [0.00000, 0.00000, 1.00000]])
    transfer_mat_post_adapt = transfer_mat_h @ np.linalg.inv(transfer_mat_cat02)
    lms_h = utils.apply_transfer_mat(lms_c, transfer_mat_post_adapt)

    lms_a = np.zeros_like(lms_h)
    lms_white_a = np.zeros_like(lms_white)
    for i in range(3):
        lms_a[:, :, i] = _cat02_nonlinearity(lms_h[:, :, i], FL)
        lms_white_a[i] = _cat02_nonlinearity(lms_white[i], FL)

    return lms_a, lms_white_a


def _cat02_nonlinearity(lms_channel: Union[np.ndarray, float], FL: float) -> Union[np.ndarray, float]:
    '''
    Post-adaptation Michaelis-Menten equation.

    Args:
        lms_channel: L, M, or S channel input value.
        FL: Luminance level adaptation factor.

    Returns:
        Nonlinear response output.
    '''
    return np.sign(lms_channel) * 400 / (27.3 * np.power(FL * np.abs(lms_channel) / 100, -0.42) + 1) + 0.1


def hdrucs_cat(xyz: np.ndarray) -> np.ndarray:
    '''
    Color Adaptation Transform used by HDR-UCS (aka Jzazbz)

    Args:
        xyz: Tristimulus XYZ image to adapt.

    Returns:
        Adapted LMS responses for image.
    '''
    # Setting color adaptation parameters
    x_factor = 1.15
    y_factor = 0.66

    xyz_unif = np.zeros_like(xyz)
    xyz_unif[:, :, 0] = x_factor * xyz[:, :, 0] - (x_factor - 1) * xyz[:, :, 2]
    xyz_unif[:, :, 1] = y_factor * xyz[:, :, 1] - (y_factor - 1) * xyz[:, :, 0]
    xyz_unif[:, :, 2] = xyz[:, :, 2]

    transfer_mat_lms = np.array([[0.4147897, 0.579999, 0.0146480], [-0.2015100, 1.120649, 0.0531008], [-0.0166008, 0.264800, 0.6684799]])
    lms = utils.apply_transfer_mat(xyz_unif, transfer_mat_lms)

    lms_a = np.zeros_like(lms)
    for i in range(3):
        lms_a[:, :, i] = _hdrucs_nonlinearity(lms[:, :, i])

    return lms_a


def hdrucs_inverse_cat(lms_a: np.ndarray) -> np.ndarray:
    '''
    Inverse of Color Adaptation Transform used by HDR-UCS (aka Jzazbz)

    Args:
        lms_a: Adapted LMS responses for image.

    Returns:
        Tristimulus XYZ image before adaptation.
    '''
    lms = _hdrucs_inverse_nonlinearity(lms_a)

    transfer_mat_lms = np.array([[0.4147897, 0.579999, 0.0146480], [-0.2015100, 1.120649, 0.0531008], [-0.0166008, 0.264800, 0.6684799]])
    xyz_unif = utils.apply_transfer_mat(lms, np.linalg.inv(transfer_mat_lms))
    xyz = np.zeros_like(xyz_unif)

    # Setting color adaptation parameters
    x_factor = 1.15
    y_factor = 0.66

    xyz[..., 2] = xyz_unif[..., 2]
    xyz[..., 0] = (xyz_unif[..., 0] + (x_factor - 1)*xyz[..., 2]) / x_factor
    xyz[..., 1] = (xyz_unif[..., 1] + (y_factor - 1)*xyz[..., 0]) / y_factor

    return xyz


# The non-linearity used in HDR UCS is almost identical to the PQ OETF
def _hdrucs_nonlinearity(x: Union[np.ndarray, float]) -> Union[np.ndarray, float]:
    '''
    Nonlinear function used in HDR-UCS' CAT.

    Args:
        x: Input to nonlinear function.

    Returns:
        Nonlinear response to input.
    '''
    c1 = 3424 / (1 << 12)
    c2 = 2413 / (1 << 7)
    c3 = 2392 / (1 << 7)
    n = 2610 / (1 << 14)
    p = 1.7 * 2523 / (1 << 5)

    x = np.clip(x/1e4, 1e-8, None)
    return nonlinearities.five_param_nonlinearity(x, [c1, c2, c3, n, p])


def _hdrucs_inverse_nonlinearity(x: Union[np.ndarray, float]) -> Union[np.ndarray, float]:
    '''
    Inverse of nonlinear function used in HDR-UCS' CAT.

    Args:
        x: Observed nonlinear response.

    Returns:
        Output of inverse nonlinear function.
    '''
    c1 = 3424 / (1 << 12)
    c2 = 2413 / (1 << 7)
    c3 = 2392 / (1 << 7)
    n = 2610 / (1 << 14)
    p = 1.7 * 2523 / (1 << 5)
    x = np.clip(x, 0, 1)
    return 1e4 * nonlinearities.inverse_five_param_nonlinearity(x, [c1, c2, c3, n, p])


# CAT16 color adaptation
# Ref "Comprehensive color solutions: CAM16, CAT16, and CAM16-UCS", C. Li, Z. Li, Z. Wang, Y. Xu, M. R. Luo, G. Cui, M. Melgosa, M. H. Brill, and M. Pointer
def cat16(xyz: np.ndarray, xyz_white: np.ndarray, D: float, FL: float) -> Tuple[np.ndarray, np.ndarray]:
    '''
    CAT16 Color Adaption Transform

    Args:
        xyz: Tristrimulus XYZ image to adapt.
        xyz_white: White point to adapt input to.
        D: Degree of adaptation.
        FL: Luminance level adaptation factor.
    Returns:
        Adapted LMS responses for image and white point.
    '''
    transfer_mat_cat16 = np.array([[0.401288, 0.651073, -0.051461], [-0.250268, 1.204414, 0.045854], [-0.002079, 0.048952, 0.953127]])
    lms_white = transfer_mat_cat16 @ xyz_white
    # Updating transfer matrix to adapt to white point
    for i in range(3):
        transfer_mat_cat16[i, :] = (xyz_white[1]/lms_white[i] * D + 1 - D) * transfer_mat_cat16[i, :]

    lms_c = utils.apply_transfer_mat(xyz, transfer_mat_cat16)

    lms_a = np.zeros_like(lms_c)
    lms_white_a = np.zeros_like(lms_white)
    # CAT16 uses the same non-linearity as CAT02
    for i in range(3):
        lms_a[:, :, i] = _cat02_nonlinearity(lms_c[:, :, i], FL)
        lms_white_a[i] = _cat02_nonlinearity(lms_white[i], FL)

    return lms_a, lms_white_a
