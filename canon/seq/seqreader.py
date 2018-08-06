import struct
import logging
import numpy as np


class SeqReader:
    def __init__(self, seqfile=None):
        self.__seq = None
        if seqfile is not None:
            self.read_seq(seqfile)

    def read_seq(self, seqfile):
        self.__seq = SeqReader.__read_seq(seqfile)

    def get_Om(self):
        Data = self.__seq['data']
        olist = []
        for data in Data:
            imgn = int(data['image_num'])
            nindex = int(data['nindex'])
            ax = data['ax']
            ay = data['ay']
            az = data['az']
            bx = data['bx']
            by = data['by']
            bz = data['bz']
            cx = data['cx']
            cy = data['cy']
            cz = data['cz']
            olist.append([imgn, nindex, ax, bx, cx, ay, by, cy, az, bz, cz])
        return olist

    def get_Zmap(self, key, selection='nindex', thres=20):
        data = self.__seq['data']
        imgn_list = set([d['image_num'] for d in data])
        xstage = [int(d['xstage']) for d in data]
        ystage = [int(d['ystage']) for d in data]
        z_key = [float(d[key]) for d in data]
        nindex = [int(d[selection]) for d in data]
        zipped_data = zip(xstage, ystage, z_key, nindex)
        x_list = list(sorted(set(xstage)))
        y_list = list(sorted(set(ystage)))
        x_step = int(x_list[1] - x_list[0])
        y_step = int(y_list[1] - y_list[0])
        # print("X_min = {:f}, X_max = {:f}".format(min(x_list), max(x_list)))
        # print("Y_min = {:f}, Y_max = {:f}".format(min(y_list), max(y_list)))
        NX = len(x_list)
        NY = len(y_list)
        x_map = dict(zip(x_list, range(NX)))
        y_map = dict(zip(y_list, range(NY)))
        Z = np.empty((NY, NX)).astype('float32')
        N = np.zeros((NY, NX))
        Z[:, :] = np.nan
        for x, y, z, n in zipped_data:
            ix = x_map[x]
            iy = y_map[y]
            if N[iy, ix] < n and n >=thres:
                # if z < 0:
                #    z += 180
                Z[iy, ix] = z
                N[iy, ix] = n

        logging.debug("number of images: {:d}".format(len(imgn_list)))
        logging.debug("NX: {:d} x NY: {:d}    step size: {:d} x {:d}".format(NX, NY, x_step, y_step))
        return Z, N

    @staticmethod
    def merge_Zmap(z1, z2, n1, n2):
        NX_1 = len(z1[0, :])
        NY_1 = len(z1[:, 0])
        NX_2 = len(z2[0, :])
        NY_2 = len(z2[:, 0])
        NX = min([NX_1, NX_2])
        NY = min([NY_1, NY_2])
        N = np.zeros((NY, NX))
        Z = (-1.5) * np.ones((NY, NX))

        for ix in range(NX):
            for iy in range(NY):
                if n1[iy, ix] >= n2[iy, ix]:
                    Z[iy, ix] = z1[iy, ix]
                    N[iy, ix] = n1[iy, ix]
                else:
                    Z[iy, ix] = z2[iy, ix]
                    N[iy, ix] = n2[iy, ix]

        return Z, N

    @staticmethod
    def __read_seq(seqfile):
        with open(seqfile, 'rb') as seq:
            numimages, = struct.unpack('i', seq.read(4))
            result = {'numimages': numimages, 'data': []}
            for i in range(numimages):
                data = SeqReader.__read_img_data(seq)
                result['data'].append(data)
        return result

    @staticmethod
    def __read_img_data(seq):
        data = dict()
        # int(analauto(i).image_num,2),
        data['image_num'], = struct.unpack('h', seq.read(2))
        # int(analauto(i).grainindice,2),
        data['grainindice'], = struct.unpack('h', seq.read(2))
        # sngl(analauto(i).xstage),
        # sngl(analauto(i).ystage),
        data['xstage'], data['ystage'] = struct.unpack('ff', seq.read(8))
        data['x'], data['y'] = data['xstage'], data['ystage']
        # sngl(analauto(i).det_distance),
        struct.unpack('f', seq.read(4))
        # sngl(analauto(i).xcent),
        # sngl(analauto(i).ycent),
        data['xcent'], data['ycent'] = struct.unpack('ff', seq.read(8))
        # sngl(analauto(i).det_pitch),
        # sngl(analauto(i).det_yaw),
        # sngl(analauto(i).det_roll),
        struct.unpack('fff', seq.read(12))
        # int(analauto(i).nindex,2),
        data['nindex'], = struct.unpack('h', seq.read(2))
        # sngl(analauto(i).dev1),
        # sngl(analauto(i).dev2),
        # sngl(analauto(i).pixdev),
        struct.unpack('fff', seq.read(12))
        # int(analauto(i).strnindex,2),
        struct.unpack('h', seq.read(2))
        # sngl(analauto(i).str_dev1),
        # sngl(analauto(i).str_dev2),
        # sngl(analauto(i).str_pixdev),
        struct.unpack('fff', seq.read(12))
        # sngl(analauto(i).dev_eps11),
        # sngl(analauto(i).dev_eps12),
        # sngl(analauto(i).dev_eps13),
        # sngl(analauto(i).dev_eps22),
        # sngl(analauto(i).dev_eps23),
        # sngl(analauto(i).dev_eps33),
        struct.unpack('ffffff', seq.read(24))
        # sngl(analauto(i).dev_epsxx),
        # sngl(analauto(i).dev_epsxy),
        # sngl(analauto(i).dev_epsxz),
        # sngl(analauto(i).dev_epsyy),
        # sngl(analauto(i).dev_epsyz),
        # sngl(analauto(i).dev_epszz),
        struct.unpack('ffffff', seq.read(24))
        # sngl(analauto(i).dev_str11),
        # sngl(analauto(i).dev_str12),
        # sngl(analauto(i).dev_str13),
        # sngl(analauto(i).dev_str22),
        # sngl(analauto(i).dev_str23),
        # sngl(analauto(i).dev_str33),
        struct.unpack('ffffff', seq.read(24))
        # sngl(analauto(i).dev_strxx),
        # sngl(analauto(i).dev_strxy),
        # sngl(analauto(i).dev_strxz),
        # sngl(analauto(i).dev_stryy),
        # sngl(analauto(i).dev_stryz),
        # sngl(analauto(i).dev_strzz),
        struct.unpack('ffffff', seq.read(24))
        # sngl(analauto(i).vonmises),
        data['vonmises'], = struct.unpack('f', seq.read(4))
        # sngl(analauto(i).ax),
        # sngl(analauto(i).ay),
        # sngl(analauto(i).az),
        # sngl(analauto(i).bx),
        # sngl(analauto(i).by),
        # sngl(analauto(i).bz),
        # sngl(analauto(i).cx),
        # sngl(analauto(i).cy),
        # sngl(analauto(i).cz),
        data['ax'], data['ay'], data['az'], = struct.unpack('fff', seq.read(12))
        data['bx'], data['by'], data['bz'], = struct.unpack('fff', seq.read(12))
        data['cx'], data['cy'], data['cz'], = struct.unpack('fff', seq.read(12))
        # sngl(analauto(i).astx),
        # sngl(analauto(i).asty),
        # sngl(analauto(i).astz),
        # sngl(analauto(i).bstx),
        # sngl(analauto(i).bsty),
        # sngl(analauto(i).bstz),
        # sngl(analauto(i).cstx),
        # sngl(analauto(i).csty),
        # sngl(analauto(i).cstz),
        struct.unpack('fffffffff', seq.read(36))
        # sngl(analauto(i).orsnr),
        # sngl(analauto(i).orsirxy),
        # sngl(analauto(i).orsirxz),
        # sngl(analauto(i).orsiryz),
        data['orsnr___'], = struct.unpack('f', seq.read(4))
        data['orsirxy'], = struct.unpack('f', seq.read(4))
        data['orsirxz'], = struct.unpack('f', seq.read(4))
        data['orsiryz'], = struct.unpack('f', seq.read(4))
        # analauto(i).sum_peak_int,
        data['sum_peak_int'], = struct.unpack('d', seq.read(8))
        # analauto(i).aver_intensity,
        data['aver_intensity'], = struct.unpack('d', seq.read(8))
        # analauto(i).aver_background,
        struct.unpack('d', seq.read(8))
        # analauto(i).izero,
        struct.unpack('d', seq.read(8))
        # analauto(i).fluoa,
        # analauto(i).fluob,
        # analauto(i).fluoc,
        # analauto(i).fluod,
        # analauto(i).fluoe,
        struct.unpack('ddddd', seq.read(40))
        # sngl(analauto(i).exposure),
        struct.unpack('f', seq.read(4))
        # sngl(analauto(i).mrss),
        struct.unpack('f', seq.read(4))
        # sngl(analauto(i).delta1),
        # sngl(analauto(i).delta2),
        struct.unpack('ff', seq.read(8))
        # sngl(analauto(i).averpeakwidth),
        struct.unpack('f', seq.read(4))
        # sngl(analauto(i).tot_strxx),
        # sngl(analauto(i).tot_strxy),
        # sngl(analauto(i).tot_strxz),
        # sngl(analauto(i).tot_stryy),
        # sngl(analauto(i).tot_stryz),
        # sngl(analauto(i).tot_strzz),
        struct.unpack('ffffff', seq.read(24))
        # sngl(analauto(i).hydr_stress),
        struct.unpack('f', seq.read(4))
        # sngl(analauto(i).dilat_strain),
        struct.unpack('f', seq.read(4))
        # sngl(analauto(i).tot_princip_stress1),
        # sngl(analauto(i).tot_princip_stress2),
        # sngl(analauto(i).tot_princip_stress3),
        struct.unpack('fff', seq.read(12))
        # sngl(analauto(i).eq_stress),
        struct.unpack('f', seq.read(4))
        # sngl(analauto(i).eq_strain),
        struct.unpack('f', seq.read(4))
        # sngl(analauto(i).usr11),
        # sngl(analauto(i).usr12),
        # sngl(analauto(i).usr13),
        # sngl(analauto(i).usr21),
        # sngl(analauto(i).usr22),
        # sngl(analauto(i).usr23),
        # sngl(analauto(i).usr31),
        # sngl(analauto(i).usr32),
        # sngl(analauto(i).usr33),
        struct.unpack('fffffffff', seq.read(36))
        # sngl(analauto(i).uasx),
        # sngl(analauto(i).uasy),
        # sngl(analauto(i).uasz),
        # sngl(analauto(i).ubsx),
        # sngl(analauto(i).ubsy),
        # sngl(analauto(i).ubsz),
        # sngl(analauto(i).ucsx),
        # sngl(analauto(i).ucsy),
        # sngl(analauto(i).ucsz),
        struct.unpack('fffffffff', seq.read(36))
        # analauto(i).defdx,
        # analauto(i).defdy,
        # analauto(i).defdz,
        struct.unpack('ddd', seq.read(24))
        # analauto(i).compdata5,
        struct.unpack('d', seq.read(8))
        # sngl(analauto(i).uasu),
        # sngl(analauto(i).uasv),
        # sngl(analauto(i).uasw),
        # sngl(analauto(i).ubsu),
        # sngl(analauto(i).ubsv),
        # sngl(analauto(i).ubsw),
        # sngl(analauto(i).ucsu),
        # sngl(analauto(i).ucsv),
        # sngl(analauto(i).ucsw),
        struct.unpack('fffffffff', seq.read(36))
        # sngl(analauto(i).sasu),
        # sngl(analauto(i).sasv),
        # sngl(analauto(i).sasw),
        # sngl(analauto(i).sbsu),
        # sngl(analauto(i).sbsv),
        # sngl(analauto(i).sbsw),
        # sngl(analauto(i).scsu),
        # sngl(analauto(i).scsv),
        # sngl(analauto(i).scsw),
        struct.unpack('fffffffff', seq.read(36))
        # sngl(analauto(i).quatw),
        # sngl(analauto(i).quatx),
        # sngl(analauto(i).quaty),
        # sngl(analauto(i).quatz),
        struct.unpack('ffff', seq.read(16))
        # sngl(analauto(i).rodr1),
        # sngl(analauto(i).rodr2),
        # sngl(analauto(i).rodr3),
        # sngl(analauto(i).rotangle),
        struct.unpack('ffff', seq.read(16))
        # sngl(analauto(i).euler_phi),
        # sngl(analauto(i).euler_theta),
        # sngl(analauto(i).euler_psi),
        struct.unpack('fff', seq.read(12))
        # sngl(analauto(i).misor_angle),
        # sngl(analauto(i).mis_vecx),
        # sngl(analauto(i).mis_vecy),
        # sngl(analauto(i).mis_vecz),
        struct.unpack('ffff', seq.read(16))
        # sngl(analauto(i).cosangR),
        # sngl(analauto(i).cosangG),
        # sngl(analauto(i).cosangB),
        struct.unpack('fff', seq.read(12))
        # sngl(analauto(i).unused19),
        struct.unpack('f', seq.read(4))
        # sngl(analauto(i).unused20)
        struct.unpack('f', seq.read(4))
        return data