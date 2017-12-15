import matplotlib.pyplot as plt
from prospect.io import read_results
import os
import numpy as np
from scipy.ndimage import gaussian_filter as norm_kde
from dynesty.plotting import _hist2d as hist2d

par_name_translator = {
                       'logmass': r'log(M/M$_{\odot}$)',
                       'dust2': r'$\tau_\mathrm{dust}$',
                       'logtau': r'log($\tau_{\mathrm{SFH}}$)',
                       'tage': r'age',
                       'logzsol': 'log(Z/Z$_{\odot}$)',
                       'dust_index': r'$\delta_{\mathrm{attenuation}}$'
                      } 

colors = {
          'prior': 'black',
          'truth': 'purple',
          'mean': 'red'
         }

def do_all():
    """load all data, make all plots
    """
    plot_folder = 'plots/paperplots/'
    results_folder = 'results/'
    ex1_plot(plot_folder,results_folder)
    ex2_plot(plot_folder,results_folder)
    ex3_plot(plot_folder,results_folder)

def load_data(folder,example_number):
    """loads sampling results
    """
    # search output folder
    # for output matching example number
    files = [f for f in os.listdir(folder) if f.split('_')[0][-1] == str(example_number)]  
    times = [f.split('_')[-2] for f in files]
    mcmc_file, model_file = [folder+'example'+str(example_number)+'_'+max(times) + f for f in ["_mcmc.h5","_model"]]
    
    # load output, return
    res, _, mod = read_results.results_from(mcmc_file, model_file=model_file)

    return res, mod

def plot_pdf(ax,samples,weights,alpha=0.5,label='posterior',plot_mean=True,color='k',zorder=0):
    """plot posterior from fits
    """
    # smoothing routine from dynesty
    bins = int(round(10. / 0.02))
    n, b = np.histogram(samples, bins=bins, weights=weights, range=[samples.min(),samples.max()])
    n = norm_kde(n, 10.)
    x0 = 0.5 * (b[1:] + b[:-1])
    y0 = n
    ax.fill_between(x0, y0/y0.max(), color=color, alpha=alpha, label=label, zorder=zorder)

    # plot mean
    if plot_mean:
        ax.axvline(np.average(samples, weights=weights), color=colors['mean'],lw=1.5,label='<posterior>')

def plot_prior(ax,prior,nsamp=1000):
    """plot prior by using information from Prospector prior object
    """
    
    # sample PDF at regular intervals
    parsamp = np.linspace(prior.range[0],prior.range[1],nsamp)
    priorsamp = prior.distribution.pdf(parsamp,*prior.args,loc=prior.loc, scale=prior.scale)

    # plot prior
    ax.plot(parsamp,priorsamp/priorsamp.max(),color=colors['prior'],lw=1.5,linestyle='--',label='prior')

    return prior.range

def ex1_plot(plot_folder,results_folder):
    """plot all parameter priors and posteriors
    """

    # load data
    res,mod = load_data(results_folder,1)

    # generate figure
    fig, ax = plt.subplots(1,5, figsize=(13.5, 3))
    for i, name in enumerate(res['theta_labels']):

        # plot posterior
        plot_pdf(ax[i],res['chain'][:,i],res['weights'])

        # plot prior
        prior_range = plot_prior(ax[i],mod._config_dict[name]['prior'])

        # show truths
        ax[i].axvline(res['obs']['true_params'][name], color=colors['truth'],lw=1.5,label='truth')

        # limits and labels
        ax[i].set_xlim(prior_range)
        ax[i].set_ylim(0,1.1)
        ax[i].set_xlabel(par_name_translator[name])

        if i != 0:
            ax[i].set_yticklabels([])
        else:
            ax[i].set_ylabel('PDF')
            ax[i].legend(loc=3, prop={'size':8}, scatterpoints=1,fancybox=True)

    plt.tight_layout(w_pad=0.0)
    fig.subplots_adjust(wspace=0.2)
    plt.savefig(plot_folder+'example1.png',dpi=150)
    plt.close()

    # also show dust--metallicity degeneracy
    fig, ax = plt.subplots(1,1, figsize=(3, 3))
    # use 1,2,3 sigma levels
    levels = 1.0 - np.exp(-0.5 * np.arange(1, 3.1, 1) ** 2)
    met_idx, dust_idx = res['theta_labels'].index('logzsol'), res['theta_labels'].index('dust2')
    hist2d(res['chain'][:,met_idx], res['chain'][:,dust_idx], weights=res['weights'], 
           ax=ax, color='0.2',levels=levels)

    # labels and lines
    ax.set_xlabel(par_name_translator['logzsol'])
    ax.set_ylabel(par_name_translator['dust2'])
    ax.axvline(res['obs']['true_params']['logzsol'], color=colors['truth'],lw=1.5,label='truth')
    ax.axhline(res['obs']['true_params']['dust2'], color=colors['truth'],lw=1.5,label='truth')
    plt.tight_layout()
    plt.savefig(plot_folder+'example1_degen.png',dpi=150)
    plt.close()    

def ex2_plot(plot_folder,results_folder):

    # load data
    res1,mod1 = load_data(results_folder,1)
    res2,mod2 = load_data(results_folder,2)
    kwargs1 = {'label':'fit1','color':'k','alpha':0.5,'zorder':-5}
    kwargs2 = {'label':'fit2','color':'red','alpha':0.5,'zorder':0}

    # generate figure
    fig, ax = plt.subplots(1,6, figsize=(16.5, 3))
    for i, name in enumerate(res2['theta_labels']):

        # plot posteriors
        if i != 5:
            plot_pdf(ax[i],res1['chain'][:,i],res1['weights'],plot_mean=False,**kwargs1)
        plot_pdf(ax[i],res2['chain'][:,i],res2['weights'],plot_mean=False,**kwargs2)

        # plot prior
        prior_range = plot_prior(ax[i],mod2._config_dict[name]['prior'])

        # show truths
        ax[i].axvline(res2['obs']['true_params'].get(name,0.0), color=colors['truth'],lw=1.5,label='truth')

        # limits and labels
        ax[i].set_xlim(prior_range)
        ax[i].set_ylim(0,1.1)
        ax[i].set_xlabel(par_name_translator[name])

        if i != 0:
            ax[i].set_yticklabels([])
        else:
            ax[i].set_ylabel('PDF')
            ax[i].legend(loc=3, prop={'size':10}, scatterpoints=1,fancybox=True)

    plt.tight_layout(w_pad=0.0)
    fig.subplots_adjust(wspace=0.2)
    plt.savefig(plot_folder+'example2.png',dpi=150)
    plt.close()

def ex3_plot(plot_folder,results_folder):

    # load data
    res,mod = load_data(results_folder,3)

    # generate figure
    fig, ax = plt.subplots(1,5, figsize=(13.5, 3))
    for i, name in enumerate(res['theta_labels']):

        # plot posteriors
        plot_pdf(ax[i],res['chain'][:,i],res['weights'],plot_mean=True)

        # plot prior
        prior_range = plot_prior(ax[i],mod._config_dict[name]['prior'])

        # show truths
        if name == 'logtau' or name == 'tage':
            ax[i].text(0.03,0.93,'no truth',transform=ax[i].transAxes,color=colors['truth'],ha='left')
        else:
            ax[i].axvline(res['obs']['true_params'].get(name,0.0), color=colors['truth'],lw=1.5,label='truth')

        # limits and labels
        if name == 'logzsol':
            ax[i].set_xlim(-2,prior_range[1])
            ax[i].plot([-1,-1],[0,1],color=colors['prior'],lw=1.5,linestyle='--')
        else:
            ax[i].set_xlim(prior_range)
        ax[i].set_ylim(0,1.1)
        ax[i].set_xlabel(par_name_translator[name])

        if i != 0:
            ax[i].set_yticklabels([])
        else:
            ax[i].set_ylabel('PDF')
            ax[i].legend(loc=3, prop={'size':8}, scatterpoints=1,fancybox=True)

    plt.tight_layout(w_pad=0.0)
    fig.subplots_adjust(wspace=0.2)
    plt.savefig(plot_folder+'example3.png',dpi=150)
    plt.close()