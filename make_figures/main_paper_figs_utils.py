import numpy as np
import mdtraj as md
from scipy.spatial.distance import jensenshannon
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable


def convert(arr):
    b = np.empty((len(arr[0]), len(arr)))
    for n, i in enumerate(arr):
        b[:, n] = i

    bav = np.average(b, axis=1)
    bstd = np.std(b, axis=1) / np.sqrt(len(b[0, :]))

    return b, bav, bstd


def best_hummer_q(traj, native, heavy_pairs):
    BETA_CONST = 50  # 1/nm
    LAMBDA_CONST = 1.8
    NATIVE_CUTOFF = 0.45  # nanometers

    # now compute these distances for the whole trajectory
    r = md.compute_distances(traj, heavy_pairs)
    # and recompute them for just the native state
    r0 = md.compute_distances(native[0], heavy_pairs)

    q = np.mean(1.0 / (1 + np.exp(BETA_CONST * (r - LAMBDA_CONST * r0))), axis=1)
    return q


def weighted_rmsf(traj, ref, weights):
    '''Compute RMSF of ensembles with non-uniform weights'''
    weights /= np.sum(weights)  # normalize weights
    traj = traj.superpose(ref[0])  # align trajectory to reference
    # compute weighted average atomic positions
    xyz0 = np.average(traj.xyz, axis=0, weights=weights)
    rmsf = np.zeros(np.shape(traj.xyz)[1])  # set vector for rmsf
    for t in range(np.shape(traj.xyz)[0]):
        xyzt = traj.xyz[t, :, :]
        norm = np.linalg.norm(xyzt - xyz0, axis=1)
        rmsf += weights[t]*norm**2
    rmsf = np.sqrt(rmsf)
    return rmsf


def hist_diff_statistics(meta_obs, exp_obs, hist, bins, start, stop):
    ds = []
    stats = {}
    mn = meta_obs.min()
    mx = meta_obs.max()

    for i in range(start, stop):
        h, _ = np.histogram(exp_obs[i], bins=bins,
                            density=True, range=(mn, mx))
        ds.append(distances(h, hist['hrw'][i]))

    for k in ds[0].keys():
        ll = [d[k] for d in ds]
        stats[k] = [np.average(ll), np.std(ll)/np.sqrt(len(ll))]

    return stats


def distances(h1, h2):
    d = {}
    h1 = normalize(h1)
    h2 = normalize(h2)

    # euclidean distance
    d['euclidean'] = np.sqrt(np.sum((h1-h2)**2))

    # chebyshev distance
    d['chebyshev'] = np.max(np.fabs(h1-h2))

    # intersection
    d['intersection'] = 1 - np.sum(np.minimum(h1, h2))

    # hellinger distance
    d['hellinger'] = 1 - np.sum(np.sqrt(np.multiply(h1, h2)))

    # matusita distance
    d['matusita'] = np.sqrt(np.sum((np.sqrt(h1) - np.sqrt(h2))**2))

    # jensen-shannon distance
    d['js'] = jensenshannon(h1, h2)

    return d


def normalize(h):
    return h/np.sum(h)


def plot_xphi_frames(cl, theta_idxs, c, outfile=None):
    norm = mpl.colors.Normalize(vmin=c.min(), vmax=c.max())
    cmap = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.coolwarm)
    cmap.set_array([])
    fig, ax = plt.subplots()
    for i in range(1, len(theta_idxs)):
        plt.plot(cl.res['phi'][i], cl.res['x2f']
                 [i], c=cmap.to_rgba(i/2), zorder=0)
        plt.plot(cl.res['phi'][i][theta_idxs[i]], cl.res['x2f']
                 [i][theta_idxs[i]], 'ok', zorder=1)
    plt.yscale('log')
    plt.xlabel(r'$\phi_{eff}$')
    plt.ylabel(r'$\chi^2_R$')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='3%', pad=0.05)
    cbar = plt.colorbar(cmap, cax=cax)
    cbar.set_label(r'Time [ns]')
    plt.tight_layout()
    if outfile:
        plt.savefig(outfile, dpi=300)
    else:
        plt.show()


def plot_xphi(cl, theta_idxs, c, ylog=True, outfig=None):

    phi, x2i, x2f = [], [], []
    for i, _ in enumerate(c):
        phi.append(cl.res['phi'][i][theta_idxs[i]])
        x2i.append(cl.res['x2i'][i][theta_idxs[i]])
        x2f.append(cl.res['x2f'][i][theta_idxs[i]])

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.plot(c[1:], phi[1:], c='k', lw=2)
    plt.xlabel('Time [ns]')
    plt.ylabel(r'$\phi_{eff}$')

    plt.subplot(1, 2, 2)
    plt.plot(c[1:], x2i[1:], c='tab:blue', lw=2, label='Prior')
    plt.plot(c[1:], x2f[1:], c='tab:red', lw=2, label='Posterior')
    if ylog:
        plt.yscale('log')
    plt.xlabel('Time [ns]')
    plt.ylabel(r'$\chi^2_R$')
    plt.legend()
    plt.tight_layout()
    if outfig:
        plt.savefig(outfig, dpi=300)
    else:
        plt.show()

    return phi, x2i, x2f


def plot_saxs_fits(ax, data, exp, w, w0, outfig=None):
    """
    Plots the SAXS intensities and residuals (experimental, prior and reweighted) of three selected frames.

    Input
    -----
    frames (list): list with 3 frame indices to be plotted
    thidx (list): list of optimal theta indices corresponding to each plotted frame
    w0 (np.array or np.ndarray): prior weights
    dt (float): time (in time_unit) separating each experimental frame
    xlog (bool): plot x-axis in log-scale
    outfig (str): path to output figure to be saved
    """

    prior = np.average(data, weights=w0, axis=0)  # constant prior
    # fig = plt.figure(figsize=(15, 7))

    posterior = np.average(data, weights=w, axis=0)

    ax.errorbar(exp[:, 0] * 10, exp[:, 1], yerr=exp[:, 2], fmt='o', color='w', ecolor='k',
                markeredgecolor='k', ms=5, label='Experiment', zorder=0, )
    # skip the first one because it's the BME label index
    ax.plot(exp[:, 0] * 10, prior[1:], lw=0.75,
            label='Prior', c='tab:blue', zorder=1)
    ax.plot(exp[:, 0] * 10, posterior[1:], lw=0.75,
            label='Posterior', c='tab:red', zorder=2)
    ax.set_yscale('log')
    ax.set_xscale('log')

    # ax.tight_layout()

#     ax2.plot(exp[:,0] * 10, (exp[:, 1] - prior[1:]) / exp[:, 2], lw=2, label='Prior', c='tab:blue', zorder=1)
#     ax2.plot(exp[:,0] * 10, (exp[:, 1] - posterior[1:]) / exp[:, 2], lw=2, label='Posterior', c='tab:red',
#              zorder=2)

#     ax2.set_xlabel(r'q [nm$^{-1}$]')
#     if xlog:
#         ax2.set_xscale('log')
#     if n == 0:
#         ax2.set_ylabel(r'$\Delta I/\sigma$')

#     plt.tight_layout()
    if outfig:
        plt.savefig(outfig, dpi=600)
    else:
        plt.show()


def plot_histograms_ax(obs, frames, bins, hist, eobs, avexp, obs_label, outfig, prior_dim=0, hist_ax=None,
                       scaled_max=False):
    # nrows = 2
    # ncols = 4
    x = np.linspace(obs.min()+1/bins, obs.max(), bins)
    width = (obs.max() - obs.min())/bins

    for n, i in enumerate(frames):

        if prior_dim == 0:
            j = 0
        else:
            j = i

        # histograms
        he = hist_ax.hist(eobs[i], bins=bins, density=True, label='Target', histtype='step',
                          linewidth=1, color='k', zorder=3, range=(obs.min(), obs.max()))
        hist_ax.bar(x, height=hist['hpr'][j], width=width,
                    alpha=0.6, color='tab:blue', label='Prior')
        hist_ax.bar(x, height=hist['hrw'][i], width=width,
                    alpha=0.6, color='tab:red', label='Posterior')

        # averages
        mm = max(hist['hpr'][j].max(), hist['hrw'][i].max(), he[0].max())
        # if scaled_max: mm = max(max(hist['hpr'][j]), max(hist['hrw'][j]), max(hist['hpr'][j]))
        hist_ax.vlines(avexp[i], 0, mm, color='k', lw=1,
                       zorder=4, label='Experimental')
        hist_ax.vlines(hist['avpr'][j], 0, mm, linestyle='--',
                       color='tab:blue', lw=1, zorder=4)
        hist_ax.vlines(hist['avrw'][i], 0, mm, linestyle='--',
                       color='tab:red', lw=1, zorder=4)

        # if n > 3:
        #    hist_ax.xlabel(obs_label)
        # if n == 0 or n == 4:
        #    plt.ylabel('Probability density' )

        # plt.title(f't = {round(i/2)} ns')
        # plt.legend()

    # plt.tight_layout()

    # if outfig:
    #    plt.savefig(outfig, dpi=300)
    # else:
    #    plt.show()


def plot_histograms(obs, frames, bins, hist, eobs, avexp, obs_label, outfig, prior_dim=0, with_prior=True):
    frames.sort()
    fig_width = 6.6  # inches
    if len(frames) <= 4:
        nrows, ncols = (1, len(frames))
        fig_height = 3.3/2
    elif len(frames) == 8:
        nrows = 2
        ncols = 4
        fig_height = 3.3
    else:
        return None
    plt.figure(figsize=(fig_width, fig_height))

    x = np.linspace(obs.min()+1/bins, obs.max(), bins)
    width = (obs.max() - obs.min())/bins

    for n, i in enumerate(frames):
        plt.subplot(nrows, ncols, n+1)

        if prior_dim == 0:
            j = 0
        else:
            j = i

        # histograms
        he = plt.hist(eobs[i], bins=bins, density=True, label='Experimental', histtype='step',
                      linewidth=1, color='k', zorder=3, range=(obs.min(), obs.max()))
        if with_prior:
            plt.bar(x, height=hist['hpr'][j], width=width,
                    alpha=0.6, color='tab:blue', label='Prior')
        plt.bar(x, height=hist['hrw'][i], width=width,
                alpha=0.6, color='tab:red', label='Posterior')

        # averages
        mm = max(hist['hpr'][j].max(), hist['hrw'][i].max(), he[0].max())
        plt.vlines(avexp[i], 0, mm, color='k', lw=1, zorder=4)
        if with_prior:
            plt.vlines(hist['avpr'][j], 0, mm, linestyle='--',
                       color='tab:blue', lw=1, zorder=4)
        plt.vlines(hist['avrw'][i], 0, mm, linestyle='--',
                   color='tab:red', lw=1, zorder=4)

        if len(frames) <= 4:
            plt.xlabel(obs_label)
        elif len(frames) == 8:
            if i >= frames[-4]:
                plt.xlabel(obs_label)

        if n == 0 or n == 4:
            plt.ylabel('Probability density')

        plt.title(f't = {round(i/2)} ns')
        # plt.legend()

    plt.tight_layout()

    if outfig:
        plt.savefig(outfig, dpi=300)
    else:
        plt.show()


def plot_saxs_fits(ax, data, exp, w, w0, residuals=True, res_ax=None):
    """
    Plots the SAXS intensities and residuals (experimental, prior and reweighted) of three selected frames.

    Input
    -----
    frames (list): list with 3 frame indices to be plotted
    thidx (list): list of optimal theta indices corresponding to each plotted frame
    w0 (np.array or np.ndarray): prior weights
    dt (float): time (in time_unit) separating each experimental frame
    xlog (bool): plot x-axis in log-scale
    outfig (str): path to output figure to be saved
    """

    prior = np.average(data, weights=w0, axis=0)
    # fig = plt.figure(figsize=(15, 7))

    posterior = np.average(data, weights=w, axis=0)

    ax.errorbar(exp[:, 0] * 10, exp[:, 1], yerr=exp[:, 2], fmt='o', color='w', ecolor='k',
                markeredgecolor='k', ms=2, label='Experiment', zorder=0, markeredgewidth=0.5)
    # skip the first one because it's the BME label index
    ax.plot(exp[:, 0] * 10, prior[1:], lw=0.75,
            label='Prior', c='tab:blue', zorder=1)
    ax.plot(exp[:, 0] * 10, posterior[1:], lw=0.75,
            label='Posterior', c='tab:red', zorder=2)
    ax.set_yscale('log')
    ax.set_xscale('log')
    # ax.set_ylabel(r'Intensity [cm$^{-1}$]')
    # ax.legend()
    # ax.tight_layout()

    if residuals:
        res_ax.plot(exp[:, 0] * 10, (exp[:, 1] - prior[1:]) /
                    exp[:, 2], lw=0.75, label='Prior', c='tab:blue', zorder=1)
        res_ax.plot(exp[:, 0] * 10, (exp[:, 1] - posterior[1:]) / exp[:, 2], lw=0.75, label='Posterior', c='tab:red',
                    zorder=2)

        res_ax.set_xlabel(r'q [nm$^{-1}$]')
        # if xlog:
        res_ax.set_xscale('log')
        # if n == 0:
        #    res_ax.set_ylabel(r'$\Delta I/\sigma$')

#     plt.tight_layout()
#     if outfig:
#         plt.savefig(outfig, dpi=300)
#     else:
    # plt.show()
