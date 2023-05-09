from typing import Union, List
import numpy as np


# Functional form of non-linearity used by PQ and PU functions.
def five_param_nonlinearity(x: Union[np.ndarray, float], params: Union[np.ndarray, List]) -> np.ndarray:
    '''
    Five-parameter non-linear function.
    
    .. math::
    
        \left(\\frac{p_0 + p_1 x^{p_3}}{1 + p_2 x^{p_3}}\\right)^{p_4}

    Args:
        x: Input (linear light) value(s)
        params: Values of five parameters that define the function

    Returns:
        Output non-linear encoded value(s)
    '''    
    assert len(params) == 5, 'Expecting only 5 parameters'
    return np.power((params[0] + params[1]*np.power(x, params[3])) / (1 + params[2]*np.power(x, params[3])), params[4])


# Functional form of inverse non-linearity used by PQ and PU functions.
def inverse_five_param_nonlinearity(y, params):
    '''
    Five-parameter inverse non-linear function.

    .. math::

        \left(\\frac{y^{1/p_4} - p_0}{p_1 - p_2 y^{1/p_4}}\\right)^{1/p_3}

    Args:
        x: Input non-linear encoded value(s)
        params: Values of five parameters that define the function

    Returns:
        Output non-linear encoded value(s)
    '''    
    assert len(params) == 5, 'Expecting only 5 parameters'
    return np.power(np.clip(np.power(y, 1/params[4]) - params[0], 0, None) / (params[1] - params[2]*np.power(y, 1/params[4])), 1/params[3])


# Opto-electrical and Electro-optical Transfer functions
def srgb_oetf(x):
    '''
    sRBG opto-electrical transfer function (linear->non-linear transformation)

    Args:
        x: Input linear light value(s) in the range [0, 1]

    Returns:
        Non-linear encoded value(s) in the range [0, 1]
    '''
    x = np.clip(x, 0, 1)
    return np.clip(np.where(x <= 0.0031308, 323*x/25, (211*np.power(x, 5/12) - 11)/200), 0, 1)


def srgb_eotf(y):
    '''
    sRBG electro-optical transfer function (non-linear->linear transformation)

    Args:
        x: Input non-linear encoded value(s) in the range [0, 1]

    Returns:
        Linear light value(s) in the range [0, 1]
    '''
    y = np.clip(y, 0, 1)
    return np.where(y < 0.04045, 25*y/323, np.power((200*y + 11)/211, 2.4))


def rec709_oetf(x):
    '''
    Rec.709 opto-electrical transfer function (linear->non-linear transformation)

    Args:
        x: Input linear light value(s) in the range [0, 1]

    Returns:
        Non-linear encoded value(s) in the range [0, 1]
    '''
    x = np.clip(x, 0, 1)
    return np.clip(np.where(x < 0.018, 4.5*x, 1.099*np.power(x, 0.45) - 0.099), 0, 1)


def rec709_eotf(y):
    '''
    Rec.709 electro-optical transfer function (non-linear->linear transformation)

    Args:
        x: Input non-linear encoded value(s) in the range [0, 1]

    Returns:
        Linear light value(s) in the range [0, 1]
    '''
    y = np.clip(y, 0, 1)
    return np.where(y < 0.081, y/4.5, np.power((y + 0.099)/1.099, 1/0.45))


def rec2020_oetf(x):
    '''
    Rec.2020 opto-electrical transfer function (linear->non-linear transformation)

    Args:
        x: Input linear light value(s) in the range [0, 1]

    Returns:
        Non-linear encoded value(s) in the range [0, 1]
    '''
    return rec709_oetf(x)  # Rec.2020 uses (almost) the same OETF as Rec.709


def rec2020_eotf(y):
    '''
    Rec.2020 electro-optical transfer function (non-linear->linear transformation)

    Args:
        x: Input non-linear encoded value(s) in the range [0, 1]

    Returns:
        Linear light value(s) in the range [0, 1]
    '''
    return rec709_eotf(y)  # Rec.2020 uses (almost) the same EOTF as Rec.709


# Often used as a substitute for OETF since display-referred values are of interest.
def pq_inv_eotf(x):
    '''
    Rec.2100 PQ inverse electro-optical transfer function (linear->non-linear transformation)

    Args:
        x: Input linear light value(s) in the range [0, 1]

    Returns:
        Non-linear encoded value(s) in the range [0, 1]
    '''
    x = np.clip(x, 1e-8, 1)
    params = np.array([0.8359375, 18.8515625, 18.6875, 0.15930175781, 78.84375])
    return five_param_nonlinearity(x, params)


def pq_eotf(y):
    '''
    Rec.2100 PQ electro-optical transfer function (non-linear->linear transformation)

    Args:
        x: Input non-linear encoded value(s) in the range [0, 1]

    Returns:
        Linear light value(s) in the range [0, 1]
    '''
    y = np.clip(y, 0, 1)
    params = np.array([0.8359375, 18.8515625, 18.6875, 0.15930175781, 78.84375])
    return inverse_five_param_nonlinearity(y, params)


def hlg_oetf(x):
    '''
    Rec.2100 HLG opto-electrical transfer function (linear->non-linear transformation)

    Args:
        x: Input linear light value(s) in the range [0, 1]

    Returns:
        Non-linear encoded value(s) in the range [0, 1]
    '''
    x = np.clip(x, 0, 1)
    return np.where(x <= 1.0/12, np.sqrt(3 * x), 0.17883277*np.log(12*x - 0.28466892) + 0.55991072953)


def hlg_eotf(y):
    '''
    Rec.2100 HLG electro-optical transfer function (non-linear->linear transformation)

    Args:
        x: Input non-linear encoded value(s) in the range [0, 1]

    Returns:
        Linear light value(s) in the range [0, 1]
    '''
    y = np.clip(y, 0, 1)
    return np.where(y <= 1.0/2, y**2 / 3, (np.exp((y - 0.55991072953) / 0.17883277) + 0.28466892) / 12)


def pu21_oetf(x):
    '''
    PU21 opto-electrical transfer function (linear->non-linear transformation)

    Args:
        x: Input linear light value(s) in the range [0, 1e4]

    Returns:
        Non-linear encoded value(s)
    '''
    x = np.clip(x, 5e-7, None)
    params = np.array([1.0000, 0.6543, 0.3283, 0.3674, 1.1107, 1.0495, 384.9215])
    return np.clip(params[6]*(five_param_nonlinearity(x, params[:5]) - params[5]), 0, None)


def pu21_eotf(y):
    '''
    PU21 electro-optical transfer function (non-linear->linear transformation)

    Args:
        x: Input non-linear encoded value(s)

    Returns:
        Linear light value(s) in the range [0, 1e4]
    '''
    params = np.array([1.0000, 0.6543, 0.3283, 0.3674, 1.1107, 1.0495, 384.9215])
    return inverse_five_param_nonlinearity(y / params[6] + params[5], params[:5])


# Primarily for convenience
def _identity(x):
    '''
    Identity function

    Args:
        x: Input value(s).

    Returns:
        Input value(s) unchanged.
    '''
    return x


oetf_dict = {'sRGB': srgb_oetf,
             'rec.709': rec709_oetf,
             'rec.2020': rec2020_oetf,
             'pq': pq_inv_eotf,  # Use inverse EOTF instead of OETF.
             'hlg': hlg_oetf,
             'pu21': pu21_oetf,
             None: _identity}

eotf_dict = {'sRGB': srgb_eotf,
             'rec.709': rec709_eotf,
             'rec.2020': rec2020_eotf,
             'pq': pq_eotf,
             'hlg': hlg_eotf,
             'pu21': pu21_oetf,
             None: _identity}
