import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as interpolate

import mindspore as ms

import os

class DatasetWaveInverse:
    def __init__(self, batch_res, batch_bcs):
        self.batch_res = batch_res
        self.batch_bcs = batch_bcs

    def init_data(self, toTensor=True, isplot=False):
        global os
        nx = 100  # number of nodes along x axis. used here to remove the specfem's absorbing regions from PINN's computational domain
        nz = 100

        n_abs = 10  # # of nodes for absorbing B.C in both directions from specfem
        n_absx = n_abs  # nodes from left side of the domain
        n_absz = n_abs  # the top boundary is not absorbing

        ax_spec = 1.5  # domain size in specfem before removing absorbing regions
        az_spec = 0.5
        xsf = 1.3  # x location of all the seismometers in specfem

        dx = ax_spec / nx
        dz = az_spec / nz
        rho = 1.0
        self.ax = xsf - n_absx * dx  # dimension of the domain in the x direction for PINNs training. Note
        # we just need to remove the thickness of the absorbing B.C on the left since
        # xsf is (must be) smaller than where the right side absorbing B.C starts
        self.az = az_spec - n_absz * dz  # dimension of the domain in the z direction
        self.t_m = 0.5  # total time for PDE training.
        self.t_st = 0.1  # this is when we take the first I.C from specfem
        self.t_s = 0.5  # total time series used from the seismograms

        s_spec = 5e-5  # specfem time stepsize
        t01 = 2000 * s_spec  # initial disp. input at this time from spec
        t02 = 2300 * s_spec  # sec "initial" disp. input at this time from spec instead of enforcing initial velocity
        t_la = 5000 * s_spec  # test data for comparing specfem and trained PINNs

        n_event = 1  # number of seismic events
        n_seis = 20  # number of input seismometers from SPECFEM; if events have different
        # numbers of seismometers, you have to change the lines containing n_seis accordingly
        z0_s = self.az  # z location of the first seismometer from SPECFEM in PINN's refrence frame.Here it must
        # be in km while in SPECFEM it's in meters. Note here we assume seismometers are
        # NOT all on the surface and they are on a vertical line with the same x; the first
        # seismometers is at the surface and the next one goes deeper

        zl_s = 0.06 - n_absz * dz  # z location of the last seismometer at depth. this doesn't have
        # to be zero and can be higher especially if you have absorbing B.C at the bottom, change
        # this accordingly based on what you used from specfem

        self.Lx = 3  # this is for scaling the wavespeed in the PDE via saling x coordinate
        self.Lz = 3  # this is for scaling the wavespeed in the PDE via scaling z coordinate

        ub = np.array([self.ax / self.Lx, self.az / self.Lz, (self.t_m - self.t_st)]).reshape(-1, 1).T  # normalization of the input to the NN
        ub0 = np.array([self.ax / self.Lx, self.az / self.Lz]).reshape(-1, 1).T  # same for the inverse NN estimating the wave_speed

        ### PDE residuals
        print('batch_size', ':', self.batch_res)
        X_pde = np.random.normal(size=(self.batch_res, 3))
        X_pde[:, 0] = X_pde[:, 0] * self.ax / self.Lx
        X_pde[:, 1] = X_pde[:, 1] * self.az / self.Lz
        X_pde[:, 2] = X_pde[:, 2] * (self.t_m - self.t_st)

        ###initial conditions for all events
        X0 = np.loadtxt(
            'event1/wavefields/wavefield_grid_for_dumps_000.txt')  # coordinates on which the wavefield output is recorded on specfem. It's the same for all the runs with the same meshing system in specfem

        X0 = X0 / 1000  # specfem works with meters unit so we need to convert them to Km
        X0[:, 0:1] = X0[:, 0:1] / self.Lx  # scaling the spatial domain
        X0[:, 1:2] = X0[:, 1:2] / self.Lz  # scaling the spatial domain
        xz = np.concatenate((X0[:, 0:1], X0[:, 1:2]), axis=1)

        n_ini = 40

        xx, zz = np.meshgrid(np.linspace(0, self.ax / self.Lx, n_ini), np.linspace(0, self.az / self.Lz, n_ini))
        xxzz = np.concatenate((xx.reshape((-1, 1)), zz.reshape((-1, 1))), axis=1)
        X_init1 = np.concatenate(
            (xx.reshape((-1, 1)), zz.reshape((-1, 1)), 0.0 * np.ones((n_ini ** 2, 1), dtype=np.float64)),
            axis=1)  # for enforcing the disp I.C
        X_init2 = np.concatenate(
            (xx.reshape((-1, 1)), zz.reshape((-1, 1)), (t02 - t01) * np.ones((n_ini ** 2, 1), dtype=np.float64)),
            axis=1)  # for enforcing the sec I.C, another snapshot of specfem
        X_init = np.concatenate((X_init1, X_init2), axis=0)

        # interpolationg specfem results in the non-absrobing part of the domain only
        xf = n_absx * dx  # start of the nonabsorbing part of the domain in specfem
        zf = n_absz * dz
        xxs, zzs = np.meshgrid(np.linspace(xf / self.Lx, xsf / self.Lx, n_ini), np.linspace(zf / self.Lz, az_spec / self.Lz, n_ini))
        xxzzs = np.concatenate((xxs.reshape((-1, 1)), zzs.reshape((-1, 1))), axis=1)

        u_scl = 1 / 3640  # scaling the output data to cover [-1 1] interval

        import os
        # uploading the wavefields from specfem
        wfs = sorted(os.listdir('event1/wavefields/.'))
        U0 = [np.loadtxt('event1/wavefields/' + f) for f in wfs]

        U_ini1 = interpolate.griddata(xz, U0[0], xxzzs, fill_value=0.0)
        U_ini1x = U_ini1[:, 0:1] / u_scl
        U_ini1z = U_ini1[:, 1:2] / u_scl

        U_ini2 = interpolate.griddata(xz, U0[1], xxzzs, fill_value=0.0)
        U_ini2x = U_ini2[:, 0:1] / u_scl
        U_ini2z = U_ini2[:, 1:2] / u_scl

        U_spec = interpolate.griddata(xz, U0[2], xxzzs, fill_value=0.0)  # Test data
        U_specx = U_spec[:, 0:1] / u_scl
        U_specz = U_spec[:, 1:2] / u_scl

        # the first event's data has been uploaded above and below
        # the rest of the n-1 events will be added
        for ii in range(n_event - 1):
            wfs = sorted(os.listdir('event' + str(ii + 2) + '/wavefields/.'))
            U0 = [np.loadtxt('event' + str(ii + 2) + '/wavefields/' + f) for f in wfs]

            U_ini1 = interpolate.griddata(xz, U0[0], xxzzs, fill_value=0.0)
            U_ini1x += U_ini1[:, 0:1] / u_scl
            U_ini1z += U_ini1[:, 1:2] / u_scl

            U_ini2 = interpolate.griddata(xz, U0[1], xxzzs, fill_value=0.0)
            U_ini2x += U_ini2[:, 0:1] / u_scl
            U_ini2z += U_ini2[:, 1:2] / u_scl

            U_spec = interpolate.griddata(xz, U0[2], xxzzs, fill_value=0.0)
            U_specx += U_spec[:, 0:1] / u_scl
            U_specz += U_spec[:, 1:2] / u_scl
        # U_ini=U_ini.reshape(-1,1)

        if isplot:
            ################### plots of inputs for sum of the events
            fig = plt.figure()
            plt.contourf(xx * self.Lx, zz * self.Lz, np.sqrt(U_ini1x ** 2 + U_ini1z ** 2).reshape(xx.shape), 100, cmap='jet')
            plt.xlabel('x')
            plt.ylabel('z')
            plt.title('Scaled I.C total disp. input specfem t=' + str(t01))
            plt.colorbar()
            plt.axis('scaled')
            plt.savefig('Ini_total_disp_spec_sumEvents.png', dpi=400)
            plt.show()
            plt.close(fig)

            fig = plt.figure()
            plt.contourf(xx * self.Lx, zz * self.Lz, np.sqrt(U_ini2x ** 2 + U_ini2z ** 2).reshape(xx.shape), 100, cmap='jet')
            plt.xlabel('x')
            plt.ylabel('z')
            plt.title('Scaled sec I.C total disp. input specfem t=' + str(round(t02, 4)))
            plt.colorbar()
            plt.axis('scaled')
            plt.savefig('sec_wavefield_input_spec_sumEvents.png', dpi=400)
            plt.show()
            plt.close(fig)

            fig = plt.figure()
            plt.contourf(xx * self.Lx, zz * self.Lz, np.sqrt(U_specx ** 2 + U_specz ** 2).reshape(xx.shape), 100, cmap='jet')
            plt.xlabel('x')
            plt.ylabel('z')
            plt.title('Test data: Total displacement specfem t=' + str(round((t_la - t01), 4)))
            plt.colorbar()
            plt.axis('scaled')
            plt.savefig('total_disp_spec_testData_sumEvents.png', dpi=400)
            plt.show()
            plt.close(fig)
            ###############################################################

        ################# ----Z component seismograms
        #################input seismograms for the first event

        import os
        sms = sorted(os.listdir('event1/seismograms/.'))
        smsz = [f for f in sms if f[-6] == 'Z']  # Z cmp seismos
        seismo_listz = [np.loadtxt('event1/seismograms/' + f) for f in smsz]  # Z cmp seismos

        t_spec = -seismo_listz[0][0, 0] + seismo_listz[0][:,
                                          0]  # specfem's time doesn't start from zero for the seismos, so we shift it forward to zero
        cut_u = t_spec > self.t_s  # here we include only part of the seismograms from specfem that are within PINNs' training time domain which is [t_st t_m]
        cut_l = t_spec < self.t_st  # Cutting the seismograms to only after the time the first snapshot from specfem is used for PINNs
        l_su = len(cut_u) - sum(cut_u)  # this is the index of the time axis in specfem after which t>t_m
        l_sl = sum(cut_l)

        l_f = 100  # subsampling seismograms from specfem
        index = np.arange(l_sl, l_su, l_f)  # subsampling every l_s time steps from specfem in the training interval
        l_sub = len(index)
        t_spec_sub = t_spec[index].reshape((-1, 1))  # subsampled time axis of specfem for the seismograms

        t_spec_sub = t_spec_sub - t_spec_sub[
            0]  # shifting the time axis back to zero. length of t_spec_sub must be equal to t_m-t_st

        for ii in range(len(seismo_listz)):
            seismo_listz[ii] = seismo_listz[ii][index]

        Sz = seismo_listz[0][:, 1].reshape(-1, 1)
        for ii in range(len(seismo_listz) - 1):
            Sz = np.concatenate((Sz, seismo_listz[ii + 1][:, 1].reshape(-1, 1)), axis=0)

        #################################################################
        #######input seismograms for the rest of the events added to the first event

        for ii in range(n_event - 1):
            sms = sorted(os.listdir('event' + str(ii + 2) + '/seismograms/.'))
            smsz = [f for f in sms if f[-6] == 'Z']  # Z cmp seismos
            seismo_listz = [np.loadtxt('event' + str(ii + 2) + '/seismograms/' + f) for f in smsz]

            for jj in range(len(seismo_listz)):
                seismo_listz[jj] = seismo_listz[jj][index]

            Sze = seismo_listz[0][:, 1].reshape(-1, 1)
            for jj in range(len(seismo_listz) - 1):
                Sze = np.concatenate((Sze, seismo_listz[jj + 1][:, 1].reshape(-1, 1)), axis=0)

            Sz += Sze
        ###########################################################

        Sz = Sz / u_scl  # scaling the sum of all seismogram inputs

        # X_S is the training collection of input coordinates in space-time for all seismograms
        X_S = np.empty([int(np.size(Sz)), 3])

        d_s = np.abs((zl_s - z0_s)) / (n_seis - 1)  # the distance between seismometers

        for i in range(len(seismo_listz)):
            X_S[i * l_sub:(i + 1) * l_sub, ] = np.concatenate((self.ax / self.Lx * np.ones((l_sub, 1), dtype=np.float64),
                                                               (z0_s - i * d_s) / self.Lz * np.ones((l_sub, 1),
                                                                                               dtype=np.float64),
                                                               t_spec_sub), axis=1)

        ################# ----X component seismograms
        #################input seismograms for the first event

        import os
        sms = sorted(os.listdir('event1/seismograms/.'))
        smsx = [f for f in sms if f[-6] == 'X']  # X cmp seismos
        seismo_listx = [np.loadtxt('event1/seismograms/' + f) for f in smsx]  # X cmp seismos

        for ii in range(len(seismo_listx)):
            seismo_listx[ii] = seismo_listx[ii][index]

        Sx = seismo_listx[0][:, 1].reshape(-1, 1)
        for ii in range(len(seismo_listx) - 1):
            Sx = np.concatenate((Sx, seismo_listx[ii + 1][:, 1].reshape(-1, 1)), axis=0)

        #################################################################
        #######input seismograms for the rest of the events added to the first event

        for ii in range(n_event - 1):
            sms = sorted(os.listdir('event' + str(ii + 2) + '/seismograms/.'))
            smsx = [f for f in sms if f[-6] == 'X']  # X cmp seismos
            seismo_listx = [np.loadtxt('event' + str(ii + 2) + '/seismograms/' + f) for f in smsx]

            for jj in range(len(seismo_listx)):
                seismo_listx[jj] = seismo_listx[jj][index]

            Sxe = seismo_listx[0][:, 1].reshape(-1, 1)
            for jj in range(len(seismo_listx) - 1):
                Sxe = np.concatenate((Sxe, seismo_listx[jj + 1][:, 1].reshape(-1, 1)), axis=0)

            Sx += Sxe
        ###########################################################

        Sx = Sx / u_scl  # scaling the sum of all seismogram inputs

        ####  BCs: Free stress on top and no BC for other sides (absorbing)
        self.bcxn = 100
        self.bctn = 50
        assert self.batch_bcs == (self.bcxn * self.bctn)
        x_vec = np.random.rand(self.bcxn, 1) * self.ax / self.Lx
        t_vec = np.random.rand(self.bctn, 1) * (self.t_m - self.t_st)
        xxb, ttb = np.meshgrid(x_vec, t_vec)
        X_BC_t = np.concatenate(
            (xxb.reshape((-1, 1)), self.az / self.Lz * np.ones((xxb.reshape((-1, 1)).shape[0], 1)), ttb.reshape((-1, 1))), axis=1)

        if toTensor:
            X_pde = ms.Tensor(X_pde, dtype=ms.float32)
            X_init = ms.Tensor(X_init, dtype=ms.float32)
            X_S = ms.Tensor(X_S, dtype=ms.float32)
            X_BC_t = ms.Tensor(X_BC_t, dtype=ms.float32)
            U_ini1x = ms.Tensor(U_ini1x, dtype=ms.float32)
            U_ini1z = ms.Tensor(U_ini1z, dtype=ms.float32)
            U_ini2x = ms.Tensor(U_ini2x, dtype=ms.float32)
            U_ini2z = ms.Tensor(U_ini2z, dtype=ms.float32)
            Sx = ms.Tensor(Sx, dtype=ms.float32)
            Sz = ms.Tensor(Sz, dtype=ms.float32)

        return X_pde, X_init, X_S, X_BC_t, U_ini1x, U_ini1z, U_ini2x, U_ini2z, Sx, Sz

    def get_batch_res(self, toTensor=True):
        X_pde = np.random.normal(size=(self.batch_res, 3))
        X_pde[:, 0] = X_pde[:, 0] * self.ax / self.Lx
        X_pde[:, 1] = X_pde[:, 1] * self.az / self.Lz
        X_pde[:, 2] = X_pde[:, 2] * (self.t_m - self.t_st)

        if toTensor:
            X_pde = ms.Tensor(X_pde, dtype=ms.float32)
        return X_pde

    def get_batch_bcs(self, toTensor=True):
        x_vec = np.random.rand(self.bcxn, 1) * self.ax / self.Lx
        t_vec = np.random.rand(self.bctn, 1) * (self.t_m - self.t_st)
        xxb, ttb = np.meshgrid(x_vec, t_vec)
        X_BC_t = np.concatenate(
            (xxb.reshape((-1, 1)), self.az / self.Lz * np.ones((xxb.reshape((-1, 1)).shape[0], 1)), ttb.reshape((-1, 1))), axis=1)

        if toTensor:
            X_BC_t = ms.Tensor(X_BC_t, dtype=ms.float32)
        return X_BC_t

if __name__ == '__main__':
    ds = DatasetWaveInverse(40000, 5000)
    r = ds.init_data()
    for d in r:
        print(d.shape)

    print(ds.get_batch_res().shape)
    print(ds.get_batch_bcs().shape)
