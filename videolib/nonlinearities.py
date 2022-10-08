import numpy as np


# Functional form of non-linearity used by PQ and PU functions.
def five_param_nonlinearity(x, params):
    assert len(params) == 5, 'Expecting only 5 parameters'
    return np.power((params[0] + params[1]*np.power(x, params[3])) / (1 + params[2]*np.power(x, params[3])), params[4])


# Functional form of inverse non-linearity used by PQ and PU functions.
def inverse_five_param_nonlinearity(y, params):
    assert len(params) == 5, 'Expecting only 5 parameters'
    return np.power(np.clip(np.power(y, 1/params[4]) - params[0], 0, None) / (params[1] - params[2]*np.power(y, 1/params[4])), 1/params[3])


# Opto-electrical and Electro-optical Transfer functions
# sRBG opto-electrical transfer function (linear->non-linear transformation)
def srgb_oetf(x):
    x = np.clip(x, 0, 1)
    return np.clip(np.where(x <= 0.0031308, 323*x/25, (211*np.power(x, 5/12) - 11)/200), 0, 1)


# sRBG electro-optical transfer function (non-linear->linear transformation)
def srgb_eotf(y):
    y = np.clip(y, 0, 1)
    return np.where(y < 0.04045, 25*y/323, np.power((200*y + 11)/211, 2.4))


# Rec.709 opto-electrical transfer function (linear->non-linear transformation)
def rec709_oetf(x):
    x = np.clip(x, 0, 1)
    return np.clip(np.where(x < 0.018, 4.5*x, 1.099*np.power(x, 0.45) - 0.099), 0, 1)


# Rec.709 electro-optical transfer function (non-linear->linear transformation)
def rec709_eotf(y):
    y = np.clip(y, 0, 1)
    return np.where(y < 0.081, y/4.5, np.power((y + 0.099)/1.099, 1/0.45))


# Rec.2020 opto-electrical transfer function (linear->non-linear transformation)
def rec2020_oetf(x):
    return rec709_oetf(x)  # Rec.2020 uses (almost) the same OETF as Rec.709


# Rec.2020 electro-optical transfer function (non-linear->linear transformation)
def rec2020_eotf(y):
    return rec709_eotf(y)  # Rec.2020 uses (almost) the same EOTF as Rec.709


# PQ HDR inverse electro-optical transfer function (linear->non-linear transformation)
# Often used as a substitute for OETF since display-referred values are of interest.
def pq_inv_eotf(x):
    x = np.clip(x, 1e-8, 1)
    params = np.array([0.8359375, 18.8515625, 18.6875, 0.15930175781, 78.84375])
    return five_param_nonlinearity(x, params)


# PQ HDR electro-optical transfer function (non-linear->linear transformation)
def pq_eotf(y):
    y = np.clip(y, 0, 1)
    params = np.array([0.8359375, 18.8515625, 18.6875, 0.15930175781, 78.84375])
    return inverse_five_param_nonlinearity(y, params)


# HLG HDR opto-electrical transfer function (linear->non-linear transformation)
def hlg_oetf(x):
    x = np.clip(x, 0, 1)
    return np.where(x <= 1.0/12, np.sqrt(3 * x), 0.17883277*np.log(12*x - 0.28466892) + 0.55991072953)


# HLG HDR electro-optical transfer function (non-linear->linear transformation)
def hlg_eotf(y):
    y = np.clip(y, 0, 1)
    return np.where(y <= 1.0/2, y**2 / 3, (np.exp((y - 0.55991072953) / 0.17883277) + 0.28466892) / 12)


# PU21 achromatic opto-electrical transfer function (linear->non-linear transformation)
def pu21_achrom_oetf(x):
    x = np.clip(x, 5e-7, None)
    params = np.array([1.0000, 0.6543, 0.3283, 0.3674, 1.1107, 1.0495, 384.9215])
    return np.clip(params[6]*(five_param_nonlinearity(x, params[:5]) - params[5]), 0, None)


# PU21 achromatic electro-optical transfer function (non-linear->linear transformation)
def pu21_achrom_eotf(y):
    params = np.array([1.0000, 0.6543, 0.3283, 0.3674, 1.1107, 1.0495, 384.9215])
    return inverse_five_param_nonlinearity(y / params[6] + params[5], params[:5])


# PU21 red-green chromatic opto-electrical transfer function (linear->non-linear transformation)
def pu21_chrom_rg_oetf(x):
    x = np.clip(x, 1e-8, 1)
    params = np.array([1.0000, 0.6297, 0.0012, 0.7097, 0.1255, 1.0018, 597.6315])
    return five_param_nonlinearity(x, params)


# PU21 red-green chromatic electro-optical transfer function (non-linear->linear transformation)
def pu21_chrom_rg_eotf(y):
    y = np.clip(y, 0, 1)
    params = np.array([1.0000, 0.6297, 0.0012, 0.7097, 0.1255, 1.0018, 597.6315])
    return inverse_five_param_nonlinearity(y / params[6] + params[5], params[:5])


# PU21 violet-yellow chromatic opto-electrical transfer function (linear->non-linear transformation)
def pu21_chrom_vy_oetf(x):
    x = np.clip(x, 5e-3, 1)
    params = np.array([1.5934, 0.1629, 0.0423, 0.4048, 0.8130, 1.4688, 753.3745])
    return five_param_nonlinearity(x, params)


# PU21 violet-yellow chromatic electro-optical transfer function (non-linear->linear transformation)
def pu21_chrom_vy_eotf(y):
    y = np.clip(y, 0, 1)
    params = np.array([1.5934, 0.1629, 0.0423, 0.4048, 0.8130, 1.4688, 753.3745])
    return inverse_five_param_nonlinearity(y / params[6] + params[5], params[:5])


# Identity function for convenience
def identity(x):
    return x


oetf_dict = {'sRGB': srgb_oetf,
             'rec.709': rec709_oetf,
             'rec.2020': rec2020_oetf,
             'pq': pq_inv_eotf,  # Use inverse EOTF instead of OETF.
             'hlg': hlg_oetf,
             'pu21_achrom': pu21_achrom_oetf,
             None: identity}

eotf_dict = {'sRGB': srgb_eotf,
             'rec.709': rec709_eotf,
             'rec.2020': rec2020_eotf,
             'pq': pq_eotf,
             'hlg': hlg_eotf,
             'pu21_achrom': pu21_achrom_oetf,
             None: identity}
