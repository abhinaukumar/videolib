from typing import Callable, Optional, Tuple, Union
import numpy as np
from . import constants
from . import standards
from . import color_adaptation
from . import utils


# Helper functions to create and apply transfer matrices
# Calculating transfer matrices from BGR to YUV based on given transfer coefficients
def get_bgr2yuv_transfer_mat(standard: standards.Standard) -> np.ndarray:
    '''
    Calculates conversion matrix from BGR to YUV

    Args:
        standard: Standard for which matrix is to be returned

    Returns:
        3x3 conversion matrix
    '''
    if standard.name in ['sRGB', 'Rec.709']:
        kb, kg, kr = 0.0722, 0.7152, 0.2126
    else:
        kb, kg, kr = 0.0593, 0.6780, 0.2627
    return np.array([[kb, kg, kr], [0.5, -0.5*kg/(1 - kb), -0.5*kr/(1 - kb)], [-0.5*kb/(1-kr), -0.5*kg/(1-kr), 0.5]])


# Calculating transfer matrices from linearized BGR to XYZ based on the color primaries of the BGR space.
# For example, color primaries vary between Rec. 709 and Rec. 2020
# Ref: http://www.brucelindbloom.com/index.html?Eqn_RGB_XYZ_Matrix.html
def get_bgr2xyz_transfer_mat(standard: standards.Standard) -> np.ndarray:
    '''
    Calculates conversion matrix from BGR to tristimulus XYZ

    Args:
        standard: Standard for which matrix is to be returned

    Returns:
        3x3 conversion matrix
    '''
    xw, yw = standard.white_xy
    xb, yb = standard.blue_xy
    xg, yg = standard.green_xy
    xr, yr = standard.red_xy

    Xw, Yw, Zw = xw/yw, 1, (1 - xw - yw)/yw
    Xr, Yr, Zr = xr/yr, 1, (1 - xr - yr)/yr
    Xg, Yg, Zg = xg/yg, 1, (1 - xg - yg)/yg
    Xb, Yb, Zb = xb/yb, 1, (1 - xb - yb)/yb

    S_mat = np.array([[Xr, Xg, Xb], [Yr, Yg, Yb], [Zr, Zg, Zb]])
    Sr, Sg, Sb = np.linalg.inv(S_mat) @ np.array([Xw, Yw, Zw])
    transfer_mat = np.array([[Sb*Xb, Sg*Xg, Sr*Xr], [Sb*Yb, Sg*Yg, Sr*Yr], [Sb*Zb, Sg*Zg, Sr*Zr]])
    return transfer_mat


# Color conversion from YUV

# Color conversion from YUV to Linear YUV
def yuv2linear_yuv(yuv: np.ndarray, standard: standards.Standard, range: Optional[Union[int, float]] = None) -> np.ndarray:
    '''
    Converts YUV to Linear YUV.

    Args:
        yuv: YUV array of shape (R, C, 3)
        standard: Color standard to which the data conforms.
        range: Range of pixel values (Default: None - inferred from standard).

    Returns:
        Linear YUV array of shape (R, C, 3)
    '''
    if range is None:
        range = standard.range
    yuv = yuv / range
    linear_yuv = standard.eotf(yuv)
    return linear_yuv


# Color conversion from YUV to BGR.
def yuv2bgr(yuv: np.ndarray, standard: standards.Standard, range: Optional[Union[int, float]] = None) -> np.ndarray:
    '''
    Converts YUV to BGR.

    Args:
        yuv: YUV array of shape (R, C, 3)
        standard: Color standard to which the data conforms.
        range: Range of pixel values (Default: None - inferred from standard).

    Returns:
        BGR array of shape (R, C, 3)
    '''
    if range is None:
        range = standard.range
    transfer_mat = np.linalg.inv(get_bgr2yuv_transfer_mat(standard))
    yuv = yuv.copy()  # Create a copy to avoid aliasing due to subsequent inplace operations
    yuv[:, :, 1] = yuv[:, :, 1] - ((range + 1) >> 1)
    yuv[:, :, 2] = yuv[:, :, 2] - ((range + 1) >> 1)
    bgr = utils.apply_transfer_mat(yuv, transfer_mat)
    return np.clip(bgr, 0, range)


# Color conversion from YUV to Linear BGR.
def yuv2linear_bgr(yuv: np.ndarray, standard: standards.Standard, range: Optional[Union[int, float]] = None) -> np.ndarray:
    '''
    Converts YUV to BGR.

    Args:
        yuv: YUV array of shape (R, C, 3)
        standard: Color standard to which the data conforms.
        range: Range of pixel values (Default: None - inferred from standard).

    Returns:
        Linear BGR array of shape (R, C, 3)
    '''
    bgr = yuv2bgr(yuv, standard, range)
    if range is None:
        range = standard.range
    bgr = bgr / range
    linear_bgr = standard.eotf(bgr)
    return linear_bgr


# Color conversion from YUV to RGB
def yuv2rgb(yuv: np.ndarray, standard: standards.Standard, range: Optional[Union[int, float]] = None) -> np.ndarray:
    '''
    Converts YUV to RGB.
    Args:
        yuv: YUV array of shape (R, C, 3)
        standard: Color standard to which the data conforms.
        range: Range of pixel values (Default: None - inferred from standard).

    Returns:
        RGB array of shape (R, C, 3)
    '''
    if range is None:
        range = standard.range
    bgr = yuv2bgr(yuv, standard, range)
    rgb = bgr[..., [2, 1, 0]]
    return rgb


# Color conversion from YUV to Linear RGB
def yuv2linear_rgb(yuv: np.ndarray, standard: standards.Standard, range: Optional[Union[int, float]] = None) -> np.ndarray:
    '''
    Converts YUV to Linear RGB.
    Args:
        yuv: YUV array of shape (R, C, 3)
        standard: Color standard to which the data conforms.
        range: Range of pixel values (Default: None - inferred from standard).

    Returns:
        Linear RGB array of shape (R, C, 3)
    '''
    rgb = yuv2rgb(yuv, standard, range)
    if range is None:
        range = standard.range
    rgb = rgb / range
    if standard == standards.rec_2100_hlg:
        rgb = (1 - 0.00043033148)*rgb + 0.00043033148
    scene_linear_rgb = standard.eotf(rgb)
    if standard == standards.rec_2100_hlg:
        scene_linear_yuv = rgb2yuv(rgb, standard, range=1)
        linear_rgb = np.expand_dims(np.power(scene_linear_yuv[..., 0], 0.62), -1) * scene_linear_rgb
    else:
        linear_rgb = scene_linear_rgb

    return linear_rgb


# Color conversion from YUV to XYZ tristimulus.
# Performed using an intermediate conversion to BGR.
def yuv2xyz(yuv: np.ndarray, standard: standards.Standard, range: Optional[Union[int, float]] = None) -> np.ndarray:
    '''
    Converts YUV to tristimulus XYZ. Performed using an intermediate conversion to BGR.

    Args:
        yuv: YUV array of shape (R, C, 3)
        standard: Color standard to which the data conforms.
        range: Range of pixel values (Default: None - inferred from standard).

    Returns:
        XYZ array of shape (R, C, 3)
    '''
    if range is None:
        range = standard.range
    # Compute the transfer matrix from the standard's color primaries.
    bgr = yuv2bgr(yuv, standard, range)
    xyz = bgr2xyz(bgr, standard, range)
    return xyz


# Color conversion to YUV

# Color conversion from linear YUV to YUV
def linear_yuv2yuv(linear_yuv: np.ndarray, standard: standards.Standard, range: Optional[Union[int, float]] = None) -> np.ndarray:
    '''
    Converts Linear YUV to YUV.
    Args:
        linear_yuv: Linear YUV array of shape (R, C, 3)
        standard: Color standard to which the data conforms.
        range: Range of pixel values (Default: None - inferred from standard).

    Returns:
        YUV array of shape (R, C, 3)
    '''
    if range is None:
        range = standard.range
    yuv = standard.oetf(linear_yuv) * range
    return np.clip(yuv, 0, range)


# Color conversion from BGR to YUV
def bgr2yuv(bgr: np.ndarray, standard: standards.Standard, range: Optional[Union[int, float]] = None) -> np.ndarray:
    '''
    Converts BGR to YUV.
    Args:
        bgr: BGR array of shape (R, C, 3)
        standard: Color standard to which the data conforms.
        range: Range of pixel values (Default: None - inferred from standard).

    Returns:
        YUV array of shape (R, C, 3)
    '''
    if range is None:
        range = standard.range
    transfer_mat = get_bgr2yuv_transfer_mat(standard)
    yuv = utils.apply_transfer_mat(bgr, transfer_mat)
    yuv[..., 1] = yuv[..., 1] + ((range + 1) >> 1)
    yuv[..., 2] = yuv[..., 2] + ((range + 1) >> 1)

    return np.clip(yuv, 0, range)


# Color conversion from Linear BGR to YUV
def linear_bgr2yuv(linear_bgr: np.ndarray, standard: standards.Standard, range: Optional[Union[int, float]] = None) -> np.ndarray:
    '''
    Converts Linear BGR to YUV.
    Args:
        linear_bgr: Linear BGR array of shape (R, C, 3)
        standard: Color standard to which the data conforms.
        range: Range of pixel values (Default: None - inferred from standard).

    Returns:
        YUV array of shape (R, C, 3)
    '''
    if range is None:
        range = standard.range
    bgr = standard.oetf(linear_bgr) * range
    yuv = bgr2yuv(bgr, standard, range)
    return yuv


# Color conversion from RGB to YUV
def rgb2yuv(rgb: np.ndarray, standard: standards.Standard, range: Optional[Union[int, float]] = None) -> np.ndarray:
    '''
    Converts RGB to YUV.

    Args:
        rgb: RGB array of shape (R, C, 3)
        standard: Color standard to which the data conforms.
        range: Range of pixel values (Default: None - inferred from standard).

    Returns:
        RGB array of shape (R, C, 3)
    '''
    if range is None:
        range = standard.range
    bgr = rgb[..., [2, 1, 0]]
    yuv = bgr2yuv(bgr, standard, range)

    return yuv


# Color conversion from Linear RGB to YUV
def linear_rgb2yuv(linear_rgb: np.ndarray, standard: standards.Standard, range: Optional[Union[int, float]] = None) -> np.ndarray:
    '''
    Converts Linear RGB to YUV.
    Args:
        linear_rgb: Linear RGB array of shape (R, C, 3)
        standard: Color standard to which the data conforms.
        range: Range of pixel values (Default: None - inferred from standard).

    Returns:
        YUV array of shape (R, C, 3)
    '''
    if range is None:
        range = standard.range
    rgb = standard.oetf(linear_rgb) * range
    yuv = rgb2yuv(rgb, standard, range)
    return yuv


# Color conversion from XYZ tristimulus to YUV.
# Performed using an intermediate conversion to BGR.
def xyz2yuv(xyz: np.ndarray, standard: standards.Standard, range: Optional[Union[int, float]] = None) -> np.ndarray:
    '''
    Converts tristimulus XYZ to YUV. Performed using an intermediate conversion to BGR.

    Args:
        xyz: XYZ array of shape (R, C, 3)
        standard: Color standard to which the data conforms.
        range: Range of pixel values (Default: None - inferred from standard).

    Returns:
        YUV array of shape (R, C, 3)
    '''
    if range is None:
        range = standard.range
    # Compute the transfer matrix from the standard's color primaries.
    bgr = xyz2bgr(xyz, standard, range)
    yuv = bgr2yuv(bgr, standard, range)
    return yuv


# Conversions between other color spaces.

# Color conversion from BGR to XYZ tristimulus.
def bgr2xyz(bgr: np.ndarray, standard: standards.Standard, range: Optional[Union[int, float]] = None) -> np.ndarray:
    '''
    Converts BGR to tristimulus XYZ.

    Args:
        bgr: BGR array of shape (R, C, 3)
        standard: Color standard to which the data conforms.
        range: Range of pixel values (Default: None - inferred from standard).

    Returns:
        XYZ array of shape (R, C, 3)
    '''
    if range is None:
        range = standard.range
    # Compute the transfer matrix from the standard's color primaries.
    transfer_mat = get_bgr2xyz_transfer_mat(standard)
    bgr = bgr / range
    linear_bgr = standard.eotf(bgr)
    xyz = utils.apply_transfer_mat(linear_bgr, transfer_mat)
    return xyz


# Color conversion from XYZ tristimulus to BGR.
def xyz2bgr(xyz: np.ndarray, standard: standards.Standard, range: Optional[Union[int, float]] = None) -> np.ndarray:
    '''
    Converts tristimulus XYZ to BGR.

    Args:
        xyz: XYZ array of shape (R, C, 3)
        standard: Color standard to which the data conforms.
        range: Range of pixel values (Default: None - inferred from standard).

    Returns:
        BGR array of shape (R, C, 3)
    '''
    if range is None:
        range = standard.range
    # Compute the transfer matrix from the standard's color primaries.
    transfer_mat = np.linalg.inv(get_bgr2xyz_transfer_mat(standard))
    linear_bgr = utils.apply_transfer_mat(xyz, transfer_mat)
    bgr = standard.oetf(linear_bgr) * range
    return bgr


# Color conversion from XYZ tristimulus to CIELAB.
def xyz2lab(xyz: np.ndarray, standard: standards.Standard, range: Optional[Union[int, float]] = None) -> np.ndarray:
    '''
    Converts tristimulus XYZ to CIELAB aka La*b*.

    Args:
        xyz: XYZ array of shape (R, C, 3)
        standard: Color standard to which the data conforms.
        range: Range of pixel values (Default: None - inferred from standard).

    Returns:
        La*b* array of shape (R, C, 3)
    '''
    def f(x: np.ndarray) -> np.ndarray:
        lim = (6/29)**3
        denom = (6/29)**2
        return np.where(x > lim, x**(1/3), x / (3*denom) + 4/29)

    white_x = standard.white_xy[0] / standard.white_xy[1]
    white_y = 1
    white_z = (1 - standard.white_xy[0] - standard.white_xy[1]) / standard.white_xy[1]
    f_x, f_y, f_z = f(xyz[..., 0] / white_x), f(xyz[..., 1] / white_y), f(xyz[..., 2] / white_z)
    L = 116 * f_y - 16
    a = 500*(f_x - f_y)
    b = 200*(f_y - f_z)
    return np.stack([L, a, b], axis=-1)


# Color conversion from CIELAB to XYZ tristimulus.
def lab2xyz(lab: np.ndarray, standard: standards.Standard, range: Optional[Union[int, float]] = None) -> np.ndarray:
    '''
    Converts CIELAB aka La*b* to tristimulus XYZ.

    Args:
        lab: La*b* array of shape (R, C, 3)
        standard: Color standard to which the data conforms.
        range: Range of pixel values (Default: None - inferred from standard).

    Returns:
        Tristimulus XYZ array of shape (R, C, 3)
    '''
    def inv_f(x: np.ndarray) -> np.ndarray:
        lim = (6/29)
        num = (6/29)**2
        return np.where(x > lim, x**3, 3*num*(x - 4/29))

    white_x = standard.white_xy[0] / standard.white_xy[1]
    white_y = 1
    white_z = (1 - standard.white_xy[0] - standard.white_xy[1]) / standard.white_xy[1]
    L_norm = (lab[..., 0] + 16)/116
    a_norm = lab[..., 1]/500
    b_norm = lab[..., 2]/200

    x = white_x * inv_f(L_norm + a_norm)
    y = white_y * inv_f(L_norm)
    z = white_z * inv_f(L_norm - b_norm)
    return np.stack([x, y, z], axis=-1)


# Color conversion from BGR to CIECAM-like spaces
def _bgr2ciecam_generic(bgr: np.ndarray, standard: standards.Standard, cat: Callable, env_setting: str = 'average', light_setting: str = 'default', bg_setting: str = 'default', three_channel: Union[str, None] = 'chroma_euclidean') -> Union[np.ndarray, Tuple]:
    '''
    Converts BGR to CIECAM-like spaces that differ in terms of their color adaptation transforms (CAT).

    Args:
        bgr: BGR array of shape (r, c, 3)
        standard: Color standard to which the data conforms.
        cat: Function defining the color adaptation transform.
        env_setting: String defining the environment setting.
        light_setting: String defining the lighting condition setting.
        bg_setting: String defining the background setting.
        three_channel: Values to be returned. Must be None (all values), 'chroma_euclidean', 'color_euclidean', or 'ucs_color_euclidean'.

    Returns:
        Array or tuple of CIECAM outputs.
    '''
    assert env_setting in constants.envs, 'Invalid choice of environment'
    assert light_setting in constants.lightintensities, 'Invalid choice of light intensity'
    assert bg_setting in constants.bgintensities, 'Invalid choice of background intensity '

    # Setting color adaptation parameters
    xyz_white = np.array((standard.white_xy) + (1 - np.sum(standard.white_xy),))
    xyz_white = xyz_white / xyz_white[1] * standard.linear_range
    Nc, c, F = constants.envs[env_setting]
    LA = constants.lightintensities[light_setting]
    Yb = constants.bgintensities[bg_setting]
    if Yb is None:
        Yb = LA/5  # Gray world assumption
    D = np.clip(F * (1 - np.exp(-(LA+42)/92)/3.6), 0, 1)
    k = 1.0 / (5*LA + 1)
    FL = 0.2 * k**4 * 5 * LA + 0.1 * (1 - k**4)**2 * (5*LA)**(1/3)

    # Convert BGR to tristimulus XYZ space
    xyz = standard.linear_range * bgr2xyz(bgr, standard)

    # Color adaptation
    lms_a, lms_white_a = cat(xyz, xyz_white, D, FL)

    # Computing appearance correlates

    # Red-green
    a = lms_a[:, :, 0] - 12/11 * lms_a[:, :, 1] + 1/11 * lms_a[:, :, 2]
    # Yellow-blue
    b = 1/9 * (lms_a[:, :, 0] + lms_a[:, :, 1] - 2*lms_a[:, :, 2])

    n = Yb / xyz_white[1]
    N = 0.725 * np.power(n, -0.2)

    # Achromatic response
    A = (2*lms_a[:, :, 0] + lms_a[:, :, 1] + 0.05*lms_a[:, :, 2] - 0.305) * N
    Aw = (2*lms_white_a[0] + lms_white_a[1] + 0.05*lms_white_a[2] - 0.305) * N

    z = 1.48 + np.sqrt(n)

    # Lightness
    J = 100*np.power((A / Aw), c*z)

    if three_channel == 'euclidean':
        return np.stack([J, a, b], axis=-1)

    # Hue angle
    h = np.arctan2(b, a)
    h = np.where(h < 0, h + 2*np.pi, h)  # Convert [-pi, pi] to [0, 2*pi]
    # Eccentricity
    e = 0.25 * (np.cos(h + 2) + 3.8)

    t = (5e4/13) * Nc * N * e * np.sqrt(a**2 + b**2)

    # Chroma
    C = np.power(t, 0.9) * (np.sqrt(J)/10) * np.power(1.64 - np.power(0.29, n), 0.73)

    # Colorfulness
    M = C * np.power(FL, 0.25)

    if three_channel is None:
        # Brightness
        Q = (4/c) * (np.sqrt(J)/10) * (Aw + 4) * np.power(FL, 0.25)

        # Saturation
        s = 100 * np.sqrt(M / Q)

        # Hue parameters
        h_arr = np.array([20.14, 90.0, 164.25, 237.53, 380.14]) * np.pi/180
        e_arr = np.array([0.8, 0.7, 1.0, 1.2, 0.8])
        H_arr = np.array([0.0, 100.0, 200.0, 300.0, 400.0])

        h_temp = np.where(h >= h_arr[0], h, h + 2*np.pi)

        inds = np.clip(np.searchsorted(h_arr, h_temp), 1, len(h_arr)-1)

        h_i = h_arr[inds-1]
        e_i = e_arr[inds-1]
        H_i = H_arr[inds-1]
        h_i1 = h_arr[inds]
        e_i1 = e_arr[inds]

        H = H_i + 100 / (1 + (e_i * (h_i1 - h_temp)) / (e_i1 * (h_temp - h_i)))

        # Return everything
        return a, b, h, H, C, M, A, J, Q, s

    if three_channel == 'chroma_euclidean':
        J_new = J
        a_new = C*np.cos(h)
        b_new = C*np.sin(h)
    elif three_channel == 'color_euclidean':
        J_new = J
        a_new = M*np.cos(h)
        b_new = M*np.sin(h)
    elif three_channel == 'ucs_color_euclidean':  # UCS based on CAT
        J_new = 1.7*J/(1 + 0.7*J)
        M_new = (1/0.0228)*np.log(1 + 0.0228*M)
        a_new = M_new*np.cos(h)
        b_new = M_new*np.sin(h)
    else:
        raise ValueError('Invalid option for three_channel')

    return np.stack([J_new, a_new, b_new], axis=-1)


# Color conversion from BGR to CIECAM02
def bgr2ciecam02(bgr: np.ndarray, standard: standards.Standard, env_setting: str = 'average', light_setting: str = 'default', bg_setting: str = 'default', three_channel: Union[str, None] = 'chroma_euclidean') -> Union[np.ndarray, Tuple]:
    '''
    Converts BGR to CIECAM02 color space that differ in terms of their color adaptation transforms (CAT).

    Args:
        bgr: BGR array of shape (R, C, 3)
        standard: Color standard to which the data conform.
        env_setting: String defining the environment setting.
        light_setting: String defining the lighting condition setting.
        bg_setting: String defining the background setting.
        three_channel: Values to be returned. Must be None (all values), 'chroma_euclidean', 'color_euclidean', or 'ucs_color_euclidean'.

    Returns:
        Array or tuple of CIECAM02 outputs.
    '''
    return _bgr2ciecam_generic(bgr, standard, color_adaptation.cat02, env_setting, light_setting, bg_setting, three_channel)


# Color conversion from BGR to an HDR uniform color space (aka Jzazbz)
# Ref: "Perceptually uniform color space for image signals including high dynamic range and wide gamut", M. Safdar, G. Cui, Y. J. Kim, and M. R. Luo
def bgr2hdrucs(bgr: np.ndarray, standard: standards.Standard, three_channel: Optional[str] = 'euclidean') -> Union[np.ndarray, Tuple]:
    '''
    Converts BGR to uniform color space designed for HDR (aka Jzazbz).

    Args:
        bgr: BGR array of shape (R, C, 3)
        standard: Color standard to which the data conforms.
        three_channel: Values to be returned. must be none (all values), 'chroma_euclidean', 'color_euclidean', or 'ucs_color_euclidean'.

    Returns:
        Array or tuple of HDR-UCS outputs.
    '''
    # Setting luminance adaptation parameters
    d = -0.56
    d0 = 1.6295e-11

    xyz = standard.linear_range * bgr2xyz(bgr, standard)

    # Color adaptation
    lms_a = color_adaptation.hdrucs_cat(xyz)

    # Computing appearance correlates

    # Red-green
    a = 3.524000 * lms_a[:, :, 0] - 4.066708 * lms_a[:, :, 1] + 0.542708 * lms_a[:, :, 2]
    # Yellow-blue
    b = 0.199076 * lms_a[:, :, 0] + 1.096799 * lms_a[:, :, 1] - 1.295875 * lms_a[:, :, 2]

    # Lightness
    I = 0.5 * lms_a[:, :, 0] + 0.5 * lms_a[:, :, 1]
    J = (1 + d)*I / (1 + d*I) - d0

    if three_channel is None:
        # Hue angle
        h = np.arctan2(b, a)
        h = np.where(h < 0, h + 2*np.pi, h)  # Convert [-pi, pi] to [0, 2*pi]

        # Chroma
        C = np.sqrt(a**2 + b**2)

        return a, b, h, C, J
    elif three_channel == 'euclidean':
        return np.stack([J, a, b], axis=-1)
    else:
        raise ValueError('Invalid option for three_channel')


# Color conversion from BGR to CAM16
# Ref "Comprehensive color solutions: CAM16, CAT16, and CAM16-UCS", C. Li, Z. Li, Z. Wang, Y. Xu, M. R. Luo, G. Cui, M. Melgosa, M. H. Brill, and M. Pointer
def bgr2cam16(bgr: np.ndarray, standard: standards.Standard, env_setting: str = 'average', light_setting: str = 'default', bg_setting: str = 'default', three_channel: Optional[str] = None) -> Union[np.ndarray, Tuple]:
    '''
    Converts BGR to CAM16 color space.

    Args:
        bgr: BGR array of shape (R, C, 3)
        standard: Color standard to which the data conform.
        env_setting: String defining the environment setting.
        light_setting: String defining the lighting condition setting.
        bg_setting: String defining the background setting.
        three_channel: Values to be returned. Must be None (all values), 'chroma_euclidean', 'color_euclidean', or 'ucs_color_euclidean'.

    Returns:
        Array or tuple of CAM16 outputs.
    '''
    return _bgr2ciecam_generic(bgr, standard, color_adaptation.cat16, env_setting, light_setting, bg_setting, three_channel)


# Color conversion from Linear RGB to HSV
def linear_rgb2hsv(linear_rgb: np.ndarray, standard: standards.Standard) -> np.ndarray:
    '''
    Converts Linear RGB to HSV.

    Args:
        linear_rgb: Linear RGB data.
        standard: Color standard to which the data conforms.

    Returns:
        HSV data
    '''
    x_max = linear_rgb.max(-1)
    x_min = linear_rgb.min(-1)
    v = x_max
    c = x_max - x_min
    h = np.where(
        c == 0, 0,
        np.where(
            v == linear_rgb[..., 0], 60*(linear_rgb[..., 1] - linear_rgb[..., 2])/c,
            np.where(
                v == linear_rgb[..., 1], 60*(2 + (linear_rgb[..., 2] - linear_rgb[..., 0])/c),
                np.where(
                    v == linear_rgb[..., 2], 60*(4 + (linear_rgb[..., 0] - linear_rgb[..., 1])/c), 0 
                )
            )
        )
    )
    s = np.divide(c, v, out=np.zeros_like(v), where=(v != 0))
    hsv = np.stack([h, s, v], axis=-1)
    return hsv


# Color conversion from HSV to Linear RGB
def hsv2linear_rgb(hsv: np.ndarray, standard: standards.Standard) -> np.ndarray:
    '''
    Converts HSV to Linear RGB.

    Args:
        hsv: HSV data.
        standard: Color standard to which the data conforms.

    Returns:
        Linear RGB data.
    '''
    def k(n):
        return np.mod(n + hsv[..., 0]/60, 6)

    def f(n):
        return hsv[..., 2] - hsv[..., 2]*hsv[..., 1]*np.clip(np.minimum(k(n), 4-k(n)), 0, 1)

    linear_rgb = np.stack([f(5), f(3), f(1)], axis=-1)
    return linear_rgb


# Utility function to get conversion function from source to destination color spaces
def get_conversion_function(src: str, dest: str) -> Callable:
    '''
    Selects color conversion function based on source and destination color spaces.

    Args:
        src: Source color space.
        dest: Destination color space.

    Returns:
        Conversion function from source to destination color space.

    Raises: 
        NameError: If conversion function does not exist.
    '''
    global_dict = globals()
    funct_key = src + '2' + dest
    if funct_key in global_dict:
        return global_dict[src + '2' + dest]
    else:
        raise NameError('Function to convert {} to {} does not exist.'.format(src, dest))
