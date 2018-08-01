import numpy as np
import sys
from kalmanfilter import kalmanfilter
from DWTfliter import  dwtfilter
import pickle
import dwtransform
import pylab

def bit_convert(data, maxbit):
    if (data & (1 << (maxbit -1))):
        data -= (1 << maxbit)
    return data  

def read_csi(csi_buf, nr, nc, num_tones):
        
    csi_matrix = np.zeros((3,3,114),dtype  = complex)   #create a complex array which store the CSI matrix
    
    idx = 0   
    bits_left = 16 #init bits_left. we process 16 bits at a time
    bitmask = np.uint32(( 1 << 10 ) - 1) #according to the h/w, we have 10 bit resolution 
                              #for each real and imag value of the csi matrix H
    h_data = csi_buf[idx] # get 16 bits for processing
    idx += 1
    h_data += (csi_buf[idx] << 8)
    idx += 1
    current_data = h_data & ((np.uint32(1) << 16) - 1)
    
    for k in range(num_tones): #loop for every subcarrier
        for nc_idx in range(nc): # loop for each tx antenna
            for nr_idx in range(nr): #loop for each rx antenna
                if (bits_left - 10 ) < 0:
                    h_data = csi_buf[idx]
                    idx += 1
                    h_data += (csi_buf[idx] << 8)
                    idx += 1
                    current_data += h_data << bits_left
                    bits_left += 16
                imag = current_data & bitmask
                imag = bit_convert(imag, 10)
                imag = np.complex(imag)
                imag = imag *(0+1j)
                csi_matrix[nr_idx][nc_idx][k] += imag
                
                bits_left -=10
                current_data = current_data >> 10
                
                if (bits_left - 10) < 0: # bits number less than 10, get next 16 bits
                    h_data = csi_buf[idx]
                    idx += 1
                    h_data += (csi_buf[idx] << 8)
                    idx += 1
                    current_data += h_data << bits_left
                    bits_left += 16
                
                real = current_data & bitmask
                real = bit_convert(real, 10)
                real = np.complex(real)
                csi_matrix[nr_idx][nc_idx][k] += real
                
                bits_left -= 10
                current_data = current_data >> 10
                
    return csi_matrix


def read_from_file(file_path):
    try:
        file = open(file_path,'rb')
    except Exception:
        print 'couldn\'t open file %s' %file_path
        file.close()
        sys.exit(0) 
    
    status = file.seek(0,2)
    if status != 0:
        pass # error message
        print 'Error2'
        
    len = file.tell()
    print 'file length is:%d\n' %len
    
    status = file.seek(0,0)
    if status != 0:
        pass # error message
        print status
        print 'error3'
    
    cur = 0
    ret = []
    
    endian_format = 'ieee-le' # some embedded system use big endian. for 16/32/64 system this should be all fine
    
    while cur < (len - 4):
        
        csi_matrix = {}
        field_len = np.fromfile(file, np.uint16, 1)
        if endian_format != 'ieee-le':
            field_len.dtype = '>u2'
        cur = cur + 2
        print 'Block length is: %d\n' %field_len
        
        if (cur + field_len)> len:
            break
        
        timestamp = np.fromfile(file, np.uint64, 1)
        if endian_format != 'ieee-le':
            timestamp.dtype = '>u8'
        csi_matrix['timestamp'] = timestamp
        cur = cur + 8
        print 'timestamp is %d\n' %timestamp
        
        csi_len = np.fromfile(file, np.uint16, 1)
        if endian_format != 'ieee-le':
            csi_len.dtype = '>u2'
        csi_matrix['csi_len'] = csi_len
        cur = cur + 2
        print 'csi_len is %d\n' %csi_len
          
        tx_channel = np.fromfile(file, np.uint16, 1)
        if endian_format != 'ieee-le':
            tx_channel.dtype = '>u2'
        csi_matrix['channel'] = tx_channel
        cur = cur + 2
        print 'channel is %d\n' %tx_channel
        
        err_info = np.fromfile(file, np.int8, 1)
        if endian_format != 'ieee-le':
            err_info.dtype = '>u1'
        else:
            err_info.dtype = 'u1'
        csi_matrix['err_info'] = err_info
        cur = cur + 1
        print 'err_info is %d\n' %err_info
     
        
        noise_floor = np.fromfile(file, np.int8, 1)
        if endian_format != 'ieee-le':
            noise_floor.dtype = '>u1'
        else:
            noise_floor.dtype = 'u1'
        csi_matrix['noise_floor'] = noise_floor
        cur = cur + 1
        print 'noise_floor is %d\n' %noise_floor
        
        Rate = np.fromfile(file, np.int8, 1) 
        if endian_format != 'ieee-le':
            Rate.dtype = '>u1'
        else:
            Rate.dtype = 'u1'
        csi_matrix['Rate'] = Rate
        cur = cur + 1
        print 'rate is %x\n' %Rate
        
        bandWidth = np.fromfile(file, np.int8, 1)
        if endian_format != 'ieee-le':
            bandWidth.dtype = '>u1'
        else:
            bandWidth.dtype = 'u1'
        csi_matrix['bandWidth'] = bandWidth
        cur = cur + 1
        print 'bandWidth is %d\n' %bandWidth
        
        num_tones = np.fromfile(file, np.int8, 1)
        if endian_format != 'ieee-le':
            num_tones.dtype = ">u1"
        else:
            num_tones.dtype = "u1"
        csi_matrix['num_tones'] = num_tones
        cur = cur + 1
        print 'num_tones is %d\n' %num_tones
        
        nr = np.fromfile(file, np.int8, 1)
        if endian_format != 'ieee-le':
            nr.dtype = '>u1'
        else:
            nr.dtype = 'u1'
        csi_matrix['nr'] = nr
        cur = cur + 1
        print 'nr is %d\n' %nr
        
        nc = np.fromfile(file, np.int8, 1)
        if endian_format != 'ieee-le':
            nc.dtype = '>u1'
        else:
            nc.dtype = 'u1'
        csi_matrix['nc'] = nc
        cur = cur + 1
        print 'nc is %d\n' %nc
        
        rssi = np.fromfile(file, np.int8, 1)
        if endian_format != 'ieee-le':
            rssi.dtype = '>u1'
        else:
            rssi.dtype = 'u1'
        csi_matrix['rssi'] = rssi
        cur = cur + 1
        print 'rssi is %d\n' %rssi
        
        rssi1 = np.fromfile(file, np.int8, 1)
        if endian_format != 'ieee-le':
            rssi1.dtype = '>u1'
        else:
            rssi1.dtype = 'u1'
        csi_matrix['rssi1'] = rssi1
        cur = cur + 1
        print 'rssi1 is %d\n' %rssi1
        
        rssi2 = np.fromfile(file, np.int8, 1)
        if endian_format != 'ieee-le':
            rssi2.dtype = '>u1'
        else:
            rssi2.dtype = 'u1'
        csi_matrix['rssi2'] = rssi2
        cur = cur + 1
        print 'rssi2 is %d\n' %rssi2
        
        rssi3 = np.fromfile(file, np.int8, 1) #wrong
        if endian_format != 'ieee-le':
            rssi3.dtype = '>u1'
        else:
            rssi3.dtype = 'u1'
        csi_matrix['rssi3'] = rssi3
        cur = cur + 1
        print 'rssi3 is %d\n' %rssi3
        
        payload_len = np.fromfile(file, np.int16, 1)
        if endian_format != 'ieee-le':
            payload_len.dtype = '>u2'
        csi_matrix['payload_len'] = payload_len
        cur = cur + 2
        print 'payload length is %d\n' %payload_len
        
        if csi_len > 0:
            csi_buf = np.fromfile(file, np.uint8, csi_len)
            csi = read_csi(csi_buf, nr, nc, num_tones)
            cur = cur + csi_len
            csi_matrix['csi'] = csi
#        else:
#            csi_matrix['csi'] =''
        
        if payload_len > 0:
            data_buf = np.fromfile(file, np.uint8, payload_len)
            cur = cur + payload_len
            csi_matrix['payload'] = data_buf
        else:
            csi_matrix['payload'] = 0
            
        if (cur + 420) > len:
            break
        ret.append(csi_matrix)
    
    if ret.__len__() > 1:
        ret = ret[0:(ret.__len__() - 1)]
        
    file.close()
    return ret

def complexToLatitude(matrix):
  return map(lambda x: float("%.2f" % abs(x)), matrix)

def linearInterpolation(matrix, time, u_time, u_index):
    raw, dwt, kal, ult= None, None, None, None
    for eachsubcarrier in matrix:
        Interpolation = np.interp(time, u_time, eachsubcarrier[u_index])
        Interpolation = dwtfilter(Interpolation).butterWorth()
        res_dwt = dwtfilter(Interpolation).filterOperation()
        # res_dwt = dwtransform.ReconUsingUpcoef(Interpolation)
        res_kal = kalmanfilter(Interpolation).feedback()
        res_ult = dwtfilter(res_kal).filterOperation()

        if raw is None:
            raw = np.array([Interpolation])
        else:
            raw = np.append(raw, [Interpolation], axis=0)
        if dwt is None:
            dwt = np.array([res_dwt])
        else:
            dwt = np.append(dwt, [res_dwt], axis=0)
        if kal is None:
            kal = np.array([res_kal])
        else:
            kal = np.append(kal, [res_kal], axis=0)
        if ult is None:
            ult = np.array([res_ult])
        else:
            ult = np.append(ult, [res_ult], axis=0)

    return raw, dwt, kal, ult

"""
append the matrix in each antenna pair, i.e., the total is nine,
the linear interpolation operation is to enrich the csi values,
the discrete wavalet transform fliter and kalmanfilter are 
to implement denoise operation.
"""

def date_wrapper():
    # file1 = read_from_file('./gestures/push.dat')
    file1 = read_from_file('test1.dat')
    timestamp = np.array([])
    startTime = file1[0]['timestamp']
    CSImatrix, CSImatrix1,  CSImatrix2, CSImatrix3, CSImatrix4, CSImatrix5,CSImatrix6, CSImatrix7, CSImatrix8, CSImatrix9= None, None, None, None, None, None, None, None, None,None

    for item in file1:
            if item['payload_len'][0] == 1040:
                timestamp = np.append(timestamp, (item['timestamp'] - startTime) / 1000000.0, axis=0)

                if CSImatrix1 is None:
                    CSImatrix1 = np.array([complexToLatitude(item['csi'][0][0])])
                else:
                    CSImatrix1 = np.append(CSImatrix1, [complexToLatitude(item['csi'][0][0])], axis=0)
                if CSImatrix2 is None:
                    CSImatrix2 = np.array([complexToLatitude(item['csi'][0][1])])
                else:
                    CSImatrix2 = np.append(CSImatrix2, [complexToLatitude(item['csi'][0][1])], axis=0)
                if CSImatrix3 is None:
                    CSImatrix3 = np.array([complexToLatitude(item['csi'][0][2])])
                else:
                    CSImatrix3 = np.append(CSImatrix3, [complexToLatitude(item['csi'][0][2])], axis=0)
                if CSImatrix4 is None:
                    CSImatrix4 = np.array([complexToLatitude(item['csi'][1][0])])
                else:
                    CSImatrix4 = np.append(CSImatrix4, [complexToLatitude(item['csi'][1][0])], axis=0)
                if CSImatrix5 is None:
                    CSImatrix5 = np.array([complexToLatitude(item['csi'][1][1])])
                else:
                    CSImatrix5 = np.append(CSImatrix5, [complexToLatitude(item['csi'][1][1])], axis=0)
                if CSImatrix6 is None:
                    CSImatrix6 = np.array([complexToLatitude(item['csi'][1][2])])
                else:
                    CSImatrix6 = np.append(CSImatrix6, [complexToLatitude(item['csi'][1][2])], axis=0)
                if CSImatrix7 is None:
                    CSImatrix7 = np.array([complexToLatitude(item['csi'][2][0])])
                else:
                    CSImatrix7 = np.append(CSImatrix7, [complexToLatitude(item['csi'][2][0])], axis=0)
                if CSImatrix8 is None:
                    CSImatrix8 = np.array([complexToLatitude(item['csi'][2][1])])
                else:
                    CSImatrix8 = np.append(CSImatrix8, [complexToLatitude(item['csi'][2][1])], axis=0)
                if CSImatrix9 is None:
                    CSImatrix9 = np.array([complexToLatitude(item['csi'][2][2])])
                else:
                    CSImatrix9 = np.append(CSImatrix6, [complexToLatitude(item['csi'][2][2])], axis=0)

    with open('train_samples.pkl', 'wb') as f:
        pickle.dump(CSImatrix1, f)
    unique_time, unique_index = np.unique(timestamp, return_index=True)
    time = np.arange(0, 5, 0.05)
    List = [CSImatrix1, CSImatrix2, CSImatrix3, CSImatrix4, CSImatrix5, CSImatrix6, CSImatrix7, CSImatrix8, CSImatrix9]
    matrix_raw, matrix_dwt, matrix_kal, matrix_ult = None, None, None, None
    for matrixList in List:
        matrixList = matrixList.transpose()
        matrixList = matrixList[0 : 56, :]                                       # TODO: the value 56 can be changed into 112 for 114 subcarrier circumstance
        raw, dwt, kal, ult= linearInterpolation(matrixList, time, unique_time, unique_index)

        if matrix_raw is None:
            matrix_raw = np.array(raw)
        else:
            matrix_raw = np.append(matrix_raw, raw, axis=1)
        if matrix_dwt is None:
            matrix_dwt = np.array(dwt)
        else:
            matrix_dwt = np.append(matrix_dwt, dwt, axis=1)
        if matrix_kal is None:
            matrix_kal = np.array(kal)
        else:
            matrix_kal = np.append(matrix_kal, kal, axis=1)
        if matrix_ult is None:
            matrix_ult = np.array(ult)
        else:
            matrix_ult = np.append(matrix_ult, ult, axis=1)

    # print CSImatrix [0]          # output is 56 * 900
    # print len(CSImatrix)

    return   matrix_raw, matrix_dwt, matrix_kal, matrix_ult

if __name__ == "__main__":
    date_wrapper()
  # kalmanturns = 1
  # onefilter = None
  # for eachturn in range(0,kalmanturns):
  #   if onefilter is None:
  #
  #     onefilter = kalman.feedback()
  #   else:
  #      kalman = kalmanfilter(onefilter)
  #      onefilter = kalman.feedback()
