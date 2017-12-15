import numpy as np
import os
from prospect.models import priors, sedmodel
from prospect.sources import CSPSpecBasis
from sedpy import observate
from astropy.cosmology import WMAP9

#############
# RUN_PARAMS
#############
APPS = os.getenv('APPS')
run_params = {'verbose':True,
              'debug': False,
              'outfile': 'results/example3',
              'nofork': True,
              # dynesty fitting specifications
              'nested_bound': 'multi', # bounding method
              'nested_sample': 'unif', # sampling method
              'nested_nlive_init': 200,
              'nested_nlive_batch': 200,
              'use_pool': {'propose_point': False},
              'nested_bootstrap': 0, 
              'nested_dlogz_init': 0.01,
              # Model info
              'zcontinuous': 2,
              'compute_vega_mags': False,
              'initial_disp':0.1,
              'interp_type': 'logarithmic',
              }
############
# OBS
#############
def load_obs(snr=10,**extras):

    # what are our mock parameters?
    params = {
              'logmass': 10,
              'logzsol': -1.5,
              'dust2': 0.05,
              'logtau': 1.0,
              'tage': 8
              }
    for p in params.keys(): params[p] = np.atleast_1d(params[p])

    # will need model and sps object to generate photometry
    mod = load_model(**run_params)
    sps = load_sps(**run_params)

    # we will also need filters
    obs = {}
    filters = ['sdss_u0', 'sdss_g0', 'sdss_r0', 'sdss_i0', 'twomass_J', 'twomass_H', 'twomass_Ks']
    obs['filters'] = observate.load_filters(filters)
    obs['wavelength'] = None

    # generate the photometry, add noise
    mod.params.update(params)
    spec, maggies, _ = mod.mean_model(mod.theta, obs, sps=sps)
    maggies_unc = (maggies / snr).squeeze()
    phot_mask = np.ones_like(maggies,dtype=bool)

    # finishing building obs dictionary
    obs['wave_effective'] = np.array([filt.wave_effective for filt in obs['filters']])
    obs['phot_mask'] = phot_mask
    obs['maggies'] = maggies
    obs['maggies_unc'] =  maggies_unc
    obs['spectrum'] = None
    obs['true_params'] = params
    obs['true_spec'] = spec

    return obs

##########################
# TRANSFORMATION FUNCTIONS
##########################
def load_gp(**extras):
    return None, None

def transform_logmass_to_mass(mass=None, logmass=None, **extras):
    return 10**logmass

def transform_logtau_to_tau(tau=None, logtau=None, **extras):
    return 10**logtau

#############
# MODEL_PARAMS
#############
model_params = []

###### BASIC PARAMETERS ##########
model_params.append({'name': 'zred', 'N': 1,
                        'isfree': False,
                        'init': 0.1,
                        'units': '',
                        'prior': priors.TopHat(mini=0.0, maxi=4.0)})

model_params.append({'name': 'logmass', 'N': 1,
                        'isfree': True,
                        'init': 10.0,
                        'units': 'Msun',
                        'prior': priors.TopHat(mini=6.0, maxi=12.0)})

model_params.append({'name': 'mass', 'N': 1,
                        'isfree': False,
                        'init': 1e10,
                        'depends_on': transform_logmass_to_mass,
                        'units': 'Msun',
                        'prior': priors.TopHat(mini=1e6, maxi=1e12)})

model_params.append({'name': 'logzsol', 'N': 1,
                        'isfree': True,
                        'init': -1.5,
                        'units': r'$\log (Z/Z_\odot)$',
                        'prior': priors.TopHat(mini=-2.00, maxi=0.19)})
                        
###### SFH   ########
model_params.append({'name': 'sfh', 'N': 1,
                        'isfree': False,
                        'init': 4,
                        'units': 'type',
                        'prior_function_name': None,
                        'prior_args': None})

model_params.append({'name': 'tau', 'N': 1,
                        'isfree': False,
                        'init': 10,
                        'depends_on': transform_logtau_to_tau,
                        'units': 'Gyr',
                        'prior': priors.TopHat(mini=0.1, maxi=100)})

model_params.append({'name': 'logtau', 'N': 1,
                        'isfree': True,
                        'init': 1,
                        'units': 'Gyr',
                        'prior': priors.TopHat(mini=-1.0, maxi=2.0)})

model_params.append({'name': 'tage', 'N': 1,
                        'isfree': True,
                        'init': 1.0,
                        'units': 'Gyr',
                        'prior': priors.TopHat(mini=0.01, maxi=13.6)})

########    IMF  ##############
model_params.append({'name': 'imf_type', 'N': 1,
                             'isfree': False,
                             'init': 1, #1 = chabrier
                             'units': None,
                             'prior_function_name': None,
                             'prior_args': None})

######## Dust Absorption ##############
model_params.append({'name': 'dust_type', 'N': 1,
                        'isfree': False,
                        'init': 2, # Calzetti
                        'units': 'index',
                        'prior_function_name': None,
                        'prior_args': None})

model_params.append({'name': 'dust2', 'N': 1,
                        'isfree': True,
                        'init': 0.1,
                        'units': '',
                        'prior': priors.TopHat(mini=0.0, maxi=4.0)})

###### Dust Emission ##############
model_params.append({'name': 'add_dust_emission', 'N': 1,
                        'isfree': False,
                        'init': 1,
                        'units': None,
                        'prior': None})

###### Nebular Emission ###########
model_params.append({'name': 'add_neb_emission', 'N': 1,
                        'isfree': False,
                        'init': False,
                        'units': r'log Z/Z_\odot',
                        'prior_function_name': None,
                        'prior_args': None})

####### Calibration ##########
model_params.append({'name': 'peraa', 'N': 1,
                     'isfree': False,
                     'init': False})

model_params.append({'name': 'mass_units', 'N': 1,
                     'isfree': False,
                     'init': 'mstar'})

###### Redefine SPS ######
def load_sps(**extras):
    sps = CSPSpecBasis(**extras)
    return sps

def load_model(**extras):

    # set tage_max, fix redshift
    n = [p['name'] for p in model_params]
    zred = model_params[n.index('zred')]['init']
    tuniv = WMAP9.age(zred).value
    model_params[n.index('tage')]['prior'].update(maxi=tuniv)

    return sedmodel.SedModel(model_params)
