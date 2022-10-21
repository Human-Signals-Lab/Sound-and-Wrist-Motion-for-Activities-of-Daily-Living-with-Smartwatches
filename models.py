import os

import torch
import torch.nn as nn
import librosa
import math
import numpy as np 
from utils.utils import paint, makedir
from utils.utils_pytorch import (
    get_info_params,
    get_info_layers,
    init_weights_orthogonal,
)
from utils.utils_attention import SelfAttention, TemporalAttention
import datetime
import _pickle as cPickle

import torch.nn.functional as F

__all__ = ["create"]

class StatisticsContainer(object):
    def __init__(self, statistics_path):
        self.statistics_path = statistics_path

        self.backup_statistics_path = '{}_{}.pickle'.format(
            os.path.splitext(self.statistics_path)[0], 
            datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

        self.statistics_dict = {'Trainloss': [], 'Testloss': [], 'test_f1': []}

    def append(self, iteration, statistics, data_type):
        #print(iteration)
        statistics['iteration'] = iteration
        self.statistics_dict[data_type].append(statistics)
        
    def dump(self):
        cPickle.dump(self.statistics_dict, open(self.statistics_path, 'wb'))
        cPickle.dump(self.statistics_dict, open(self.backup_statistics_path, 'wb'))
        
    def load_state_dict(self, resume_iteration):
        self.statistics_dict = cPickle.load(open(self.statistics_path, 'rb'))

        resume_statistics_dict = {'Trainloss': [], 'Testloss': [], 'test_f1': []}
        
        for key in self.statistics_dict.keys():
            for statistics in self.statistics_dict[key]:
                if statistics['iteration'] <= resume_iteration:
                    resume_statistics_dict[key].append(statistics)
                
        self.statistics_dict = resume_statistics_dict


class DFTBase(nn.Module):
    def __init__(self):
        """Base class for DFT and IDFT matrix"""
        super(DFTBase, self).__init__()

    def dft_matrix(self, n):
        (x, y) = np.meshgrid(np.arange(n), np.arange(n))
        omega = np.exp(-2 * np.pi * 1j / n)
        W = np.power(omega, x * y)
        return W

    def idft_matrix(self, n):
        (x, y) = np.meshgrid(np.arange(n), np.arange(n))
        omega = np.exp(2 * np.pi * 1j / n)
        W = np.power(omega, x * y)
        return W


class DFT(DFTBase):
    def __init__(self, n, norm):
        """Calculate DFT, IDFT, RDFT, IRDFT. 

        Args:
          n: fft window size
          norm: None | 'ortho'
        """
        super(DFT, self).__init__()

        self.W = self.dft_matrix(n)
        self.inv_W = self.idft_matrix(n)

        self.W_real = torch.Tensor(np.real(self.W))
        self.W_imag = torch.Tensor(np.imag(self.W))
        self.inv_W_real = torch.Tensor(np.real(self.inv_W))
        self.inv_W_imag = torch.Tensor(np.imag(self.inv_W))

        self.n = n
        self.norm = norm

    def dft(self, x_real, x_imag):
        """Calculate DFT of signal. 

        Args:
          x_real: (n,), signal real part
          x_imag: (n,), signal imag part

        Returns:
          z_real: (n,), output real part
          z_imag: (n,), output imag part
        """
        z_real = torch.matmul(x_real, self.W_real) - torch.matmul(x_imag, self.W_imag)
        z_imag = torch.matmul(x_imag, self.W_real) + torch.matmul(x_real, self.W_imag)

        if self.norm is None:
            pass
        elif self.norm == 'ortho':
            z_real /= math.sqrt(self.n)
            z_imag /= math.sqrt(self.n)

        return z_real, z_imag

        return z_real, z_imag

    def idft(self, x_real, x_imag):
        """Calculate IDFT of signal. 

        Args:
          x_real: (n,), signal real part
          x_imag: (n,), signal imag part

        Returns:
          z_real: (n,), output real part
          z_imag: (n,), output imag part
        """
        z_real = torch.matmul(x_real, self.inv_W_real) - torch.matmul(x_imag, self.inv_W_imag)
        z_imag = torch.matmul(x_imag, self.inv_W_real) + torch.matmul(x_real, self.inv_W_imag)

        if self.norm is None:
            z_real /= self.n
        elif self.norm == 'ortho':
            z_real /= math.sqrt(n)
            z_imag /= math.sqrt(n)

        return z_real, z_imag

    def rdft(self, x_real):
        """Calculate right DFT of signal. 

        Args:
          x_real: (n,), signal real part
          x_imag: (n,), signal imag part

        Returns:
          z_real: (n // 2 + 1,), output real part
          z_imag: (n // 2 + 1,), output imag part
        """
        n_rfft = self.n // 2 + 1
        z_real = torch.matmul(x_real, self.W_real[..., 0 : n_rfft])
        z_imag = torch.matmul(x_real, self.W_imag[..., 0 : n_rfft])

        if self.norm is None:
            pass
        elif self.norm == 'ortho':
            z_real /= math.sqrt(self.n)
            z_imag /= math.sqrt(self.n)

        return z_real, z_imag

    def irdft(self, x_real, x_imag):
        """Calculate inverse right DFT of signal. 

        Args:
          x_real: (n // 2 + 1,), signal real part
          x_imag: (n // 2 + 1,), signal imag part

        Returns:
          z_real: (n,), output real part
          z_imag: (n,), output imag part
        """
        n_rfft = self.n // 2 + 1

        flip_x_real = torch.flip(x_real, dims=(-1,))
        x_real = torch.cat((x_real, flip_x_real[..., 1 : n_rfft - 1]), dim=-1)

        flip_x_imag = torch.flip(x_imag, dims=(-1,))
        x_imag = torch.cat((x_imag, -1. * flip_x_imag[..., 1 : n_rfft - 1]), dim=-1)

        z_real = torch.matmul(x_real, self.inv_W_real) - torch.matmul(x_imag, self.inv_W_imag)

        if self.norm is None:
            z_real /= self.n
        elif self.norm == 'ortho':
            z_real /= math.sqrt(n)

        return z_real
        

class STFT(DFTBase):
    def __init__(self, n_fft=2048, hop_length=None, win_length=None, 
        window='hann', center=True, pad_mode='reflect', freeze_parameters=True):
        """Implementation of STFT with Conv1d. The function has the same output 
        of librosa.core.stft
        """
        super(STFT, self).__init__()

        assert pad_mode in ['constant', 'reflect']

        self.n_fft = n_fft
        self.center = center
        self.pad_mode = pad_mode

        # By default, use the entire frame
        if win_length is None:
            win_length = n_fft

        # Set the default hop, if it's not already specified
        if hop_length is None:
            hop_length = int(win_length // 4)

        fft_window = librosa.filters.get_window(window, win_length, fftbins=True)

        # Pad the window out to n_fft size
        fft_window = librosa.util.pad_center(fft_window, n_fft)

        # DFT & IDFT matrix
        self.W = self.dft_matrix(n_fft)

        out_channels = n_fft // 2 + 1

        self.conv_real = nn.Conv1d(in_channels=1, out_channels=out_channels, 
            kernel_size=n_fft, stride=hop_length, padding=0, dilation=1, 
            groups=1, bias=False)

        self.conv_imag = nn.Conv1d(in_channels=1, out_channels=out_channels, 
            kernel_size=n_fft, stride=hop_length, padding=0, dilation=1, 
            groups=1, bias=False)

        self.conv_real.weight.data = torch.Tensor(
            np.real(self.W[:, 0 : out_channels] * fft_window[:, None]).T)[:, None, :]
        # (n_fft // 2 + 1, 1, n_fft)

        self.conv_imag.weight.data = torch.Tensor(
            np.imag(self.W[:, 0 : out_channels] * fft_window[:, None]).T)[:, None, :]
        # (n_fft // 2 + 1, 1, n_fft)

        if freeze_parameters:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, input):
        """input: (batch_size, data_length)

        Returns:
          real: (batch_size, n_fft // 2 + 1, time_steps)
          imag: (batch_size, n_fft // 2 + 1, time_steps)
        """

        x = input[:, None, :]   # (batch_size, channels_num, data_length)

        if self.center:
            x = F.pad(x, pad=(self.n_fft // 2, self.n_fft // 2), mode=self.pad_mode)

        real = self.conv_real(x)
        imag = self.conv_imag(x)
        # (batch_size, n_fft // 2 + 1, time_steps)

        real = real[:, None, :, :].transpose(2, 3)
        imag = imag[:, None, :, :].transpose(2, 3)
        # (batch_size, 1, time_steps, n_fft // 2 + 1)

        return real, imag


def magphase(real, imag):
    mag = (real ** 2 + imag ** 2) ** 0.5
    cos = real / mag
    sin = imag / mag
    return mag, cos, sin


class ISTFT(DFTBase):
    def __init__(self, n_fft=2048, hop_length=None, win_length=None, 
        window='hann', center=True, pad_mode='reflect', freeze_parameters=True):
        """Implementation of ISTFT with Conv1d. The function has the same output 
        of librosa.core.istft
        """
        super(ISTFT, self).__init__()

        assert pad_mode in ['constant', 'reflect']

        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = window
        self.center = center
        self.pad_mode = pad_mode

        # By default, use the entire frame
        if win_length is None:
            win_length = n_fft

        # Set the default hop, if it's not already specified
        if hop_length is None:
            hop_length = int(win_length // 4)

        ifft_window = librosa.filters.get_window(window, win_length, fftbins=True)

        # Pad the window out to n_fft size
        ifft_window = librosa.util.pad_center(ifft_window, n_fft)

        # DFT & IDFT matrix
        self.W = self.idft_matrix(n_fft) / n_fft

        self.conv_real = nn.Conv1d(in_channels=n_fft, out_channels=n_fft, 
            kernel_size=1, stride=1, padding=0, dilation=1, 
            groups=1, bias=False)

        self.conv_imag = nn.Conv1d(in_channels=n_fft, out_channels=n_fft, 
            kernel_size=1, stride=1, padding=0, dilation=1, 
            groups=1, bias=False)

        
        self.conv_real.weight.data = torch.Tensor(
            np.real(self.W * ifft_window[None, :]).T)[:, :, None]
        # (n_fft // 2 + 1, 1, n_fft)

        self.conv_imag.weight.data = torch.Tensor(
            np.imag(self.W * ifft_window[None, :]).T)[:, :, None]
        # (n_fft // 2 + 1, 1, n_fft)
        
        if freeze_parameters:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, real_stft, imag_stft, length):
        """input: (batch_size, 1, time_steps, n_fft // 2 + 1)

        Returns:
          real: (batch_size, data_length)
        """

        device = next(self.parameters()).device
        batch_size = real_stft.shape[0]

        real_stft = real_stft[:, 0, :, :].transpose(1, 2)
        imag_stft = imag_stft[:, 0, :, :].transpose(1, 2)
        # (batch_size, n_fft // 2 + 1, time_steps)

        # Full stft
        full_real_stft = torch.cat((real_stft, torch.flip(real_stft[:, 1 : -1, :], dims=[1])), dim=1)
        full_imag_stft = torch.cat((imag_stft, - torch.flip(imag_stft[:, 1 : -1, :], dims=[1])), dim=1)

        # Reserve space for reconstructed waveform
        if length:
            if self.center:
                padded_length = length + int(self.n_fft)
            else:
                padded_length = length
            n_frames = min(
                real_stft.shape[2], int(np.ceil(padded_length / self.hop_length)))
        else:
            n_frames = real_stft.shape[2]
 
        expected_signal_len = self.n_fft + self.hop_length * (n_frames - 1)
        expected_signal_len = self.n_fft + self.hop_length * (n_frames - 1)
        y = torch.zeros(batch_size, expected_signal_len).to(device)

        # IDFT
        s_real = self.conv_real(full_real_stft) - self.conv_imag(full_imag_stft)

        # Overlap add
        for i in range(n_frames):
            y[:, i * self.hop_length : i * self.hop_length + self.n_fft] += s_real[:, :, i]

        ifft_window_sum = librosa.filters.window_sumsquare(self.window, n_frames,
            win_length=self.win_length, n_fft=self.n_fft, hop_length=self.hop_length)

        approx_nonzero_indices = np.where(ifft_window_sum > librosa.util.tiny(ifft_window_sum))[0]
        approx_nonzero_indices = torch.LongTensor(approx_nonzero_indices).to(device)
        ifft_window_sum = torch.Tensor(ifft_window_sum).to(device)
        
        y[:, approx_nonzero_indices] /= ifft_window_sum[approx_nonzero_indices][None, :]

        # Trim or pad to length
        if length is None:
            if self.center:
                y = y[:, self.n_fft // 2 : -self.n_fft // 2]
        else:
            if self.center:
                start = self.n_fft // 2
            else:
                start = 0

            y = y[:, start : start + length]
            (batch_size, len_y) = y.shape
            if y.shape[-1] < length:
                y = torch.cat((y, torch.zeros(batch_size, length - len_y).to(device)), dim=-1)

        return y
        
class Spectrogram(nn.Module):
    def __init__(self, n_fft=2048, hop_length=None, win_length=None, 
        window='hann', center=True, pad_mode='reflect', power=2.0, 
        freeze_parameters=True):
        """Calculate spectrogram using pytorch. The STFT is implemented with 
        Conv1d. The function has the same output of librosa.core.stft
        """
        super(Spectrogram, self).__init__()

        self.power = power

        self.stft = STFT(n_fft=n_fft, hop_length=hop_length, 
            win_length=win_length, window=window, center=center, 
            pad_mode=pad_mode, freeze_parameters=True)

    def forward(self, input):
        """input: (batch_size, 1, time_steps, n_fft // 2 + 1)

        Returns:
          spectrogram: (batch_size, 1, time_steps, n_fft // 2 + 1)
        """

        (real, imag) = self.stft.forward(input)
        # (batch_size, n_fft // 2 + 1, time_steps)

        spectrogram = real ** 2 + imag ** 2

        if self.power == 2.0:
            pass
        else:
            spectrogram = spectrogram ** (power / 2.0)

        return spectrogram


class LogmelFilterBank(nn.Module):
    def __init__(self, sr=32000, n_fft=2048, n_mels=64, fmin=50, fmax=14000, is_log=True, 
        ref=1.0, amin=1e-10, top_db=80.0, freeze_parameters=True):
        """Calculate logmel spectrogram using pytorch. The mel filter bank is 
        the pytorch implementation of as librosa.filters.mel 
        """
        super(LogmelFilterBank, self).__init__()

        self.is_log = is_log
        self.ref = ref
        self.amin = amin
        self.top_db = top_db

        self.melW = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels,
            fmin=fmin, fmax=fmax).T
        # (n_fft // 2 + 1, mel_bins)

        self.melW = nn.Parameter(torch.Tensor(self.melW))

        if freeze_parameters:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, input):
        """input: (batch_size, channels, time_steps)
        
        Output: (batch_size, time_steps, mel_bins)
        """

        # Mel spectrogram
        mel_spectrogram = torch.matmul(input, self.melW)

        # Logmel spectrogram
        if self.is_log:
            output = self.power_to_db(mel_spectrogram)
        else:
            output = mel_spectrogram

        return output


    def power_to_db(self, input):
        """Power to db, this function is the pytorch implementation of 
        librosa.core.power_to_lb
        """
        ref_value = self.ref
        log_spec = 10.0 * torch.log(torch.clamp(input, min=self.amin, max=np.inf))
        log_spec -= 10.0 * np.log(np.maximum(self.amin, ref_value))

        if self.top_db is not None:
            if self.top_db < 0:
                raise ParameterError('top_db must be non-negative')
            log_spec = torch.clamp(log_spec, min=log_spec.max() - self.top_db, max=np.inf)

        return log_spec

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        
        super(ConvBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=in_channels, 
                              out_channels=out_channels,
                              kernel_size=(3, 3), stride=(1, 1),
                              padding=(1, 1), bias=False)
                              
        self.conv2 = nn.Conv2d(in_channels=out_channels, 
                              out_channels=out_channels,
                              kernel_size=(3, 3), stride=(1, 1),
                              padding=(1, 1), bias=False)
                              
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.init_weight()
        
    def init_weight(self):
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_bn(self.bn1)
        init_bn(self.bn2)

        
    def forward(self, input, pool_size=(2, 2), pool_type='avg'):
        
        x = input
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        if pool_type == 'max':
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg':
            x = F.avg_pool2d(x, kernel_size=pool_size, stride=(2,2))
        elif pool_type == 'avg+max':
            x1 = F.avg_pool2d(x, kernel_size=pool_size)
            x2 = F.max_pool2d(x, kernel_size=pool_size)
            x = x1 + x2
        else:
            raise Exception('Incorrect argument!')
        
        return x



class FeatureExtractor(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        filter_num,
        filter_size,
        enc_num_layers,
        enc_is_bidirectional,
        dropout,
        dropout_rnn,
        activation,
        sa_div,
    ):
        super(FeatureExtractor, self).__init__()

        self.conv1 = nn.Conv2d(1, filter_num, (filter_size, 1))
        self.conv2 = nn.Conv2d(filter_num, filter_num, (filter_size, 1))
        self.conv3 = nn.Conv2d(filter_num, filter_num, (filter_size, 1))
        self.conv4 = nn.Conv2d(filter_num, filter_num, (filter_size, 1))
        self.activation = nn.ReLU() if activation == "ReLU" else nn.Tanh()

        self.dropout = nn.Dropout(dropout)
        self.rnn = nn.GRU(
            filter_num * input_dim,
            hidden_dim,
            enc_num_layers,
            bidirectional=enc_is_bidirectional,
            dropout=dropout_rnn,
        )

        self.ta = TemporalAttention(hidden_dim)
        self.sa = SelfAttention(filter_num, sa_div)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x = self.activation(self.conv3(x))
        x = self.activation(self.conv4(x))

        # apply self-attention on each temporal dimension (along sensor and feature dimensions)
        refined = torch.cat(
            [self.sa(torch.unsqueeze(x[:, :, t, :], dim=3)) for t in range(x.shape[2])],
            dim=-1,
        )
        x = refined.permute(3, 0, 1, 2)
        x = x.reshape(x.shape[0], x.shape[1], -1)

        x = self.dropout(x)
        outputs, h = self.rnn(x)

        # apply temporal attention on GRU outputs
        out = self.ta(outputs)
        return out


class Classifier(nn.Module):
    def __init__(self, hidden_dim, num_class):
        super(Classifier, self).__init__()
        self.fc = nn.Linear(hidden_dim, num_class)

    def forward(self, z):
        return self.fc(z)


class AttendDiscriminate(nn.Module):
    def __init__(
        self,
        model,
        input_dim,
        hidden_dim,
        filter_num,
        filter_size,
        enc_num_layers,
        enc_is_bidirectional,
        dropout,
        dropout_rnn,
        dropout_cls,
        activation,
        sa_div,
        num_class,
        train_mode,
        experiment,
        participant, **kwargs
    ):
        super(AttendDiscriminate, self).__init__()

        self.experiment = f"train_{experiment}" if train_mode else experiment
        self.model = model
        self.hidden_dim = hidden_dim
        self.participant = participant
        print(paint(f"[STEP 3] Creating {self.model} HAR model ..."))

        self.fe = FeatureExtractor(
            input_dim,
            hidden_dim,
            filter_num,
            filter_size,
            enc_num_layers,
            enc_is_bidirectional,
            dropout,
            dropout_rnn,
            activation,
            sa_div,
        )

        self.dropout = nn.Dropout(dropout_cls)
        self.classifier = Classifier(hidden_dim, num_class)
        self.register_buffer(
            "centers", (torch.randn(num_class, self.hidden_dim).cuda())
        )

        # do not create log directories if we are only testing the models module
        if experiment != "test_models":
            if train_mode:
                makedir(self.path_checkpoints)
                makedir(self.path_logs)
            makedir(self.path_visuals)

    def forward(self, x):

        feature = self.fe(x)
        z = feature.div(
            torch.norm(feature, p=2, dim=1, keepdim=True).expand_as(feature)
        )
        out = self.dropout(feature)
        logits = self.classifier(out)
        return z, logits

    @property
    def path_checkpoints(self):
        return f"./models/{self.experiment}/checkpoints/{self.participant}"

    @property
    def path_logs(self):
        return f"./models/{self.experiment}/logs/{self.participant}"

    @property
    def path_visuals(self):
        return f"./models/{self.experiment}/visuals/{self.participant}"


def init_layer(layer):

    if type(layer) == nn.LSTM:
        for name, param in layer.named_parameters():
            if 'weight_ih' in name:
                torch.nn.init.orthogonal_(param.data)
            elif 'weight_hh' in name:
                torch.nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
    else:
        """Initialize a Linear or Convolutional layer. """
        nn.init.xavier_uniform_(layer.weight)
 
        if hasattr(layer, 'bias'):
            if layer.bias is not None:
                layer.bias.data.fill_(0.)
            
    
def init_bn(bn):
    """Initialize a Batchnorm layer. """
    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.)



class Cnn14_AudioExtractor(nn.Module):
    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin, 
        fmax, classes_num):
        
        super(Cnn14_AudioExtractor, self).__init__()

        window = 'hann'
        center = True
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None

        # Spectrogram extractor
        self.spectrogram_extractor = Spectrogram(n_fft=window_size, hop_length=hop_size, 
            win_length=window_size, window=window, center=center, pad_mode=pad_mode, 
            freeze_parameters=True)

        # Logmel feature extractor
        self.logmel_extractor = LogmelFilterBank(sr=sample_rate, n_fft=window_size, 
            n_mels=mel_bins, fmin=fmin, fmax=fmax, ref=ref, amin=amin, top_db=top_db, 
            freeze_parameters=True)

        self.bn0 = nn.BatchNorm2d(64)

        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)
        self.conv_block5 = ConvBlock(in_channels=512, out_channels=1024)
        self.conv_block6 = ConvBlock(in_channels=1024, out_channels=2048)

        self.fc1 = nn.Linear(2048, 2048, bias=True)
        
        self.init_weight()

    def init_weight(self):
        init_bn(self.bn0)
        init_layer(self.fc1)
        # init_layer(self.fc_audioset)
 
    def forward(self, input, mixup_lambda=None):
        """
        Input: (batch_size, data_length)"""

        x = self.spectrogram_extractor(input)   # (batch_size, 1, time_steps, freq_bins)
        x = self.logmel_extractor(x)    # (batch_size, 1, time_steps, mel_bins)
        
        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)
        
        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')

        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block5(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block6(x, pool_size=(1, 1), pool_type='avg')
        conv_out = F.dropout(x, p=0.2, training=self.training)
        x = torch.mean(conv_out, dim=3)
        
        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu_(self.fc1(x))
        return x



class Audio_CNN14(nn.Module):
    def __init__(
        self,
        model,
        input_dim,
        hidden_dim,
        filter_num,
        filter_size,
        enc_num_layers,
        enc_is_bidirectional,
        dropout,
        dropout_rnn,
        dropout_cls,
        activation,
        sa_div,
        num_class,
        train_mode,
        experiment,
        participant,
        sample_rate, window_size, hop_size, mel_bins, fmin, fmax
    ):
        super(Audio_CNN14, self).__init__()

        self.experiment = f"train_{experiment}" if train_mode else experiment
        self.model = model
        self.hidden_dim = hidden_dim
        self.participant = participant
        print(paint(f"[STEP 3] Creating {self.model} HAR model ..."))

        self.audio_fe = Cnn14_AudioExtractor(sample_rate, window_size, hop_size, mel_bins, fmin, 
        fmax, 527)


        self.dropout = nn.Dropout(dropout_cls)
        self.classifier = Classifier(2048, num_class)        
        self.register_buffer(
            "centers", (torch.randn(num_class, 2048).cuda())
        )

        # do not create log directories if we are only testing the models module
        if experiment != "test_models":
            if train_mode:
                makedir(self.path_checkpoints)
                makedir(self.path_logs)
            makedir(self.path_visuals)


    def forward(self, x):
        embedding = self.audio_fe(x)

        z = embedding.div(
            torch.norm(embedding, p=2, dim=1, keepdim=True).expand_as(embedding)
        )

        out = self.dropout(embedding)
        logits = self.classifier(out)

        return z, logits

    @property
    def path_checkpoints(self):
        return f"./models/{self.experiment}/checkpoints/{self.participant}"

    @property
    def path_logs(self):
        return f"./models/{self.experiment}/logs/{self.participant}"

    @property
    def path_visuals(self):
        return f"./models/{self.experiment}/visuals/{self.participant}"




class AttendDiscriminate_MotionAudio_CNN14(nn.Module):
    def __init__(
        self,
        model,
        input_dim,
        hidden_dim,
        filter_num,
        filter_size,
        enc_num_layers,
        enc_is_bidirectional,
        dropout,
        dropout_rnn,
        dropout_cls,
        activation,
        sa_div,
        num_class,
        train_mode,
        experiment,
        participant,
        sample_rate, window_size, hop_size, mel_bins, fmin, fmax
    ):
        super(AttendDiscriminate_MotionAudio_CNN14, self).__init__()

        self.experiment = f"train_{experiment}" if train_mode else experiment
        self.model = model
        self.hidden_dim = hidden_dim
        self.participant = participant
        print(paint(f"[STEP 3] Creating {self.model} HAR model ..."))

        self.fe = FeatureExtractor(
            input_dim,
            hidden_dim,
            filter_num,
            filter_size,
            enc_num_layers,
            enc_is_bidirectional,
            dropout,
            dropout_rnn,
            activation,
            sa_div,
        )

        self.audio_fe = Cnn14_AudioExtractor(sample_rate, window_size, hop_size, mel_bins, fmin, 
        fmax, 527)

        self.sa = SelfAttention(2048+128, sa_div)

        self.dropout = nn.Dropout(dropout_cls)
        self.classifier = Classifier(2048+128, num_class)
        self.register_buffer(
            "centers", (torch.randn(num_class, 2048+128).cuda())
        )

        # do not create log directories if we are only testing the models module
        if experiment != "test_models":
            if train_mode:
                makedir(self.path_checkpoints)
                makedir(self.path_logs)
            makedir(self.path_visuals)

    def forward(self, x_motion, x_audio):
        feature = self.fe(x_motion)
        feature_audio = self.audio_fe(x_audio)

        z = feature.div(
            torch.norm(feature, p=2, dim=1, keepdim=True).expand_as(feature)
        )
        z_a = feature_audio.div(
            torch.norm(feature_audio, p=2, dim=1, keepdim=True).expand_as(feature_audio)
        )
        feature = torch.cat([feature,feature_audio], dim=-1)
        z = torch.cat([z, z_a], dim=-1)
        refined = self.sa(feature)

        out = self.dropout(refined)
        logits = self.classifier(out)
        return z, logits

    @property
    def path_checkpoints(self):
        return f"./models/{self.experiment}/checkpoints/{self.participant}"

    @property
    def path_logs(self):
        return f"./models/{self.experiment}/logs/{self.participant}"

    @property
    def path_visuals(self):
        return f"./models/{self.experiment}/visuals/{self.participant}"


class DeepConvLSTM_MotionAudio_CNN14_Concatenate(nn.Module):
    def __init__(
        self,
        model,
        input_dim,
        hidden_dim,
        filter_num,
        filter_size,
        enc_num_layers,
        enc_is_bidirectional,
        dropout,
        dropout_rnn,
        dropout_cls,
        activation,
        sa_div,
        num_class,
        train_mode,
        experiment,
        participant,
        sample_rate, window_size, hop_size, mel_bins, fmin, fmax
    ):
        super(DeepConvLSTM_MotionAudio_CNN14_Concatenate, self).__init__()

        self.experiment = f"train_{experiment}" if train_mode else experiment
        self.model = model
        self.hidden_dim = hidden_dim
        self.participant = participant
        print(paint(f"[STEP 3] Creating {self.model} HAR model ..."))

        self.fe = DeepConvLSTM_SplitV3(classes_num=num_class, acc_features=3, gyr_features = 3)

        self.audio_fe = Cnn14_AudioExtractor(sample_rate, window_size, hop_size, mel_bins, fmin, 
        fmax, 527)


        self.dropout = nn.Dropout(dropout_cls)
        self.classifier = Classifier(41728+2048, num_class)
        self.register_buffer(
            "centers", (torch.randn(num_class, 41728+2048).cuda())
        )

        # do not create log directories if we are only testing the models module
        if experiment != "test_models":
            if train_mode:
                makedir(self.path_checkpoints)
                makedir(self.path_logs)
            makedir(self.path_visuals)

    def forward(self, x_motion, x_audio):
        feature = self.fe(x_motion[:,None,:,:3], x_motion[:,None,:,3:])
        feature_audio = self.audio_fe(x_audio)

        z = feature.div(
            torch.norm(feature, p=2, dim=1, keepdim=True).expand_as(feature)
        )
        z_a = feature_audio.div(
            torch.norm(feature_audio, p=2, dim=1, keepdim=True).expand_as(feature_audio)
        )
        feature = torch.cat([feature,feature_audio], dim=-1)
        z = torch.cat([z, z_a], dim=-1)

        out = self.dropout(feature)
        logits = self.classifier(out)
        return z, logits

    @property
    def path_checkpoints(self):
        return f"./models/{self.experiment}/checkpoints/{self.participant}"

    @property
    def path_logs(self):
        return f"./models/{self.experiment}/logs/{self.participant}"

    @property
    def path_visuals(self):
        return f"./models/{self.experiment}/visuals/{self.participant}"


class DeepConvLSTM_SplitV4(nn.Module):
    def __init__(self, classes_num, acc_features, gyr_features):
        super(DeepConvLSTM_SplitV4, self).__init__()

        self.name = 'DeepConvLSTM'
        self.convAcc1 = nn.Conv2d(in_channels=1, 
                              out_channels=64,
                              kernel_size=(5, 1), stride=(1,1),
                              padding=(0,0))
        self.bnAcc1 = nn.BatchNorm2d(64)
                              
        self.convAcc2 = nn.Conv2d(in_channels=64, 
                              out_channels=64,
                              kernel_size=(5,1), stride=(1, 1),
                              padding=(0, 0))
                              
        self.bnAcc2 = nn.BatchNorm2d(64)
        self.maxPool1 = nn.MaxPool2d(kernel_size = (3,1))

        self.convAcc3 = nn.Conv2d(in_channels=64, 
                              out_channels=64,
                              kernel_size=(5, 1), stride=(1,1),
                              padding=(0,0))
        self.bnAcc3 = nn.BatchNorm2d(64)
                              
        self.convAcc4 = nn.Conv2d(in_channels=64, 
                              out_channels=64,
                              kernel_size=(5,1), stride=(1, 1),
                              padding=(0, 0))
        self.maxPool2 = nn.MaxPool2d(kernel_size = (3,1))
                          
        self.bnAcc4 = nn.BatchNorm2d(64)

        self.lstmAcc1 = nn.LSTM(64*acc_features, hidden_size=128, num_layers=1, batch_first=True)
        self.lstmAcc2 = nn.LSTM(128, 128, num_layers=1, batch_first=True)


        self.convGyr1 = nn.Conv2d(in_channels=1, 
                              out_channels=64,
                              kernel_size=(5, 1), stride=(1,1),
                              padding=(0,0))
        self.bnGyr1 = nn.BatchNorm2d(64)
                              
        self.convGyr2 = nn.Conv2d(in_channels=64, 
                              out_channels=64,
                              kernel_size=(5,1), stride=(1, 1),
                              padding=(0, 0))
                              
        self.bnGyr2 = nn.BatchNorm2d(64)

        self.convGyr3 = nn.Conv2d(in_channels=64, 
                              out_channels=64,
                              kernel_size=(5, 1), stride=(1,1),
                              padding=(0,0))
        
        self.bnGyr3 = nn.BatchNorm2d(64)
        self.maxPool3 = nn.MaxPool2d(kernel_size = (3,1))
                      
        self.convGyr4 = nn.Conv2d(in_channels=64, 
                              out_channels=64,
                              kernel_size=(5,1), stride=(1, 1),
                              padding=(0, 0))
                              
        self.bnGyr4 = nn.BatchNorm2d(64)
        self.maxPool4 = nn.MaxPool2d(kernel_size = (3,1))

        self.lstmGyr1 = nn.LSTM(64*gyr_features, hidden_size=128, num_layers=1, batch_first=True)
        self.lstmGyr2 = nn.LSTM(128, 128, num_layers=1, batch_first=True)
        self.temporalAttention_Gyr = Attention(128,128)
        self.temporalAttention_Acc = Attention(128,128)


        self.init_weight()

    def init_weight(self):
        # init_bn(self.bn0)
        init_layer(self.convAcc1)
        init_layer(self.convAcc2)
        init_layer(self.convAcc3)
        init_layer(self.convAcc4)
        init_layer(self.convGyr1)
        init_layer(self.convGyr2)
        init_layer(self.convGyr3)
        init_layer(self.convGyr4)


        init_bn(self.bnAcc1)
        init_bn(self.bnAcc2)
        init_bn(self.bnAcc3)
        init_bn(self.bnAcc4)
        init_bn(self.bnGyr1)
        init_bn(self.bnGyr2)
        init_bn(self.bnGyr3)
        init_bn(self.bnGyr4)


    def forward(self, inputAcc, inputGyr):

        x1 = self.convAcc1(inputAcc)
        x1 = self.bnAcc1(x1)
        x1 = self.convAcc2(x1)
        x1 = self.bnAcc2(x1)

        x1 = self.convAcc3(x1)
        x1 = self.bnAcc3(x1)
        x1 = self.convAcc4(x1)
        x1 = self.bnAcc4(x1)
        x1 = self.maxPool2(x1)
        
        x1 =  x1.reshape((x1.shape[0], x1.shape[2],-1)) 
        self.lstmAcc1.flatten_parameters()
        x1, _ = self.lstmAcc1(x1)
       
        self.lstmAcc2.flatten_parameters()
        x1, (h1,c1) = self.lstmAcc2(x1)
        x1, alpha1 = self.temporalAttention_Acc(x1)       

        x2 = self.convGyr1(inputGyr)
        x2 = self.bnGyr1(x2)
        x2 = self.convGyr2(x2)
        x2 = self.bnGyr2(x2)

        x2 = self.convGyr3(x2)
        x2 = self.bnGyr3(x2)
        x2 = self.convGyr4(x2)
        x2 = self.bnGyr4(x2) 
        x2 = self.maxPool4(x2)

        x2 =  x2.reshape((x2.shape[0],x2.shape[2],-1)) 
        self.lstmGyr1.flatten_parameters()
        x2, _ = self.lstmGyr1(x2)
        self.lstmGyr2.flatten_parameters()
        x2 , (h2,c2) = self.lstmGyr2(x2)
        x2, alpha2 = self.temporalAttention_Gyr(x2)        

        x3 = torch.cat([x1, x2], 1)


        return x3



class DeepConvLSTM_MotionAudio_CNN14_Attention(nn.Module):
    def __init__(
        self,
        model,
        input_dim,
        hidden_dim,
        filter_num,
        filter_size,
        enc_num_layers,
        enc_is_bidirectional,
        dropout,
        dropout_rnn,
        dropout_cls,
        activation,
        sa_div,
        num_class,
        train_mode,
        experiment,
        participant,
        sample_rate, window_size, hop_size, mel_bins, fmin, fmax
    ):
        super(DeepConvLSTM_MotionAudio_CNN14_Attention, self).__init__()

        self.experiment = f"train_{experiment}" if train_mode else experiment
        self.model = model
        self.hidden_dim = hidden_dim
        self.participant = participant
        print(paint(f"[STEP 3] Creating {self.model} HAR model ..."))

        self.fe = DeepConvLSTM_SplitV3(classes_num=num_class, acc_features=3, gyr_features = 3)

        self.audio_fe = Cnn14_AudioExtractor(sample_rate, window_size, hop_size, mel_bins, fmin, 
        fmax, 527)
        self.maxPool = nn.MaxPool1d(kernel_size = 2, stride=2)
        self.sa = SelfAttention(256+2048, sa_div)

        self.dropout = nn.Dropout(dropout_cls)
        self.classifier = Classifier(256+2048, num_class)
        self.register_buffer(
            "centers", (torch.randn(num_class, 256+2048).cuda())
        )

        # do not create log directories if we are only testing the models module
        if experiment != "test_models":
            if train_mode:
                makedir(self.path_checkpoints)
                makedir(self.path_logs)
            makedir(self.path_visuals)

    def forward(self, x_motion, x_audio):
        feature = self.fe(x_motion[:,None,:,:3], x_motion[:,None,:,3:])
        feature_audio = self.audio_fe(x_audio)

        z = feature.div(
            torch.norm(feature, p=2, dim=1, keepdim=True).expand_as(feature)
        )
        z_a = feature_audio.div(
            torch.norm(feature_audio, p=2, dim=1, keepdim=True).expand_as(feature_audio)
        )
        feature = torch.cat([feature,feature_audio], dim=-1)
        z = torch.cat([z, z_a], dim=-1)
        refined = self.sa(feature)

        out = self.dropout(refined)
        logits = self.classifier(out)
        return z, logits

    @property
    def path_checkpoints(self):
        return f"./models/{self.experiment}/checkpoints/{self.participant}"

    @property
    def path_logs(self):
        return f"./models/{self.experiment}/logs/{self.participant}"

    @property
    def path_visuals(self):
        return f"./models/{self.experiment}/visuals/{self.participant}"

class AttendDiscriminate_MotionAudio_CNN14_Concatenate(nn.Module):
    def __init__(
        self,
        model,
        input_dim,
        hidden_dim,
        filter_num,
        filter_size,
        enc_num_layers,
        enc_is_bidirectional,
        dropout,
        dropout_rnn,
        dropout_cls,
        activation,
        sa_div,
        num_class,
        train_mode,
        experiment,
        participant,
        sample_rate, window_size, hop_size, mel_bins, fmin, fmax
    ):
        super(AttendDiscriminate_MotionAudio_CNN14_Concatenate, self).__init__()

        self.experiment = f"train_{experiment}" if train_mode else experiment
        self.model = model
        self.hidden_dim = hidden_dim
        self.participant = participant
        print(paint(f"[STEP 3] Creating {self.model} HAR model ..."))

        self.fe = FeatureExtractor(
            input_dim,
            hidden_dim,
            filter_num,
            filter_size,
            enc_num_layers,
            enc_is_bidirectional,
            dropout,
            dropout_rnn,
            activation,
            sa_div,
        )

        self.audio_fe = Cnn14_AudioExtractor(sample_rate, window_size, hop_size, mel_bins, fmin, 
        fmax, 527)


        self.dropout = nn.Dropout(dropout_cls)
        self.classifier = Classifier(2048+128, num_class)
        self.register_buffer(
            "centers", (torch.randn(num_class, 2048+128).cuda())
        )

        # do not create log directories if we are only testing the models module
        if experiment != "test_models":
            if train_mode:
                makedir(self.path_checkpoints)
                makedir(self.path_logs)
            makedir(self.path_visuals)

    def forward(self, x_motion, x_audio):
        feature = self.fe(x_motion)
        feature_audio = self.audio_fe(x_audio)

        z = feature.div(
            torch.norm(feature, p=2, dim=1, keepdim=True).expand_as(feature)
        )
        z_a = feature_audio.div(
            torch.norm(feature_audio, p=2, dim=1, keepdim=True).expand_as(feature_audio)
        )
        feature = torch.cat([feature,feature_audio], dim=-1)
        z = torch.cat([z, z_a], dim=-1)

        out = self.dropout(feature)
        logits = self.classifier(out)
        return z, logits

    @property
    def path_checkpoints(self):
        return f"./models/{self.experiment}/checkpoints/{self.participant}"

    @property
    def path_logs(self):
        return f"./models/{self.experiment}/logs/{self.participant}"

    @property
    def path_visuals(self):
        return f"./models/{self.experiment}/visuals/{self.participant}"





class DeepConvLSTM_SplitV3(nn.Module):
    def __init__(self, classes_num, acc_features, gyr_features):
        super(DeepConvLSTM_SplitV3, self).__init__()

        self.name = 'DeepConvLSTM'
        self.convAcc1 = nn.Conv2d(in_channels=1, 
                              out_channels=64,
                              kernel_size=(5, 1), stride=(1,1),
                              padding=(0,0))
        self.bnAcc1 = nn.BatchNorm2d(64)
                              
        self.convAcc2 = nn.Conv2d(in_channels=64, 
                              out_channels=64,
                              kernel_size=(5,1), stride=(1, 1),
                              padding=(0, 0))
                              
        self.bnAcc2 = nn.BatchNorm2d(64)
        self.maxPool1 = nn.MaxPool2d(kernel_size = (3,1))

        self.convAcc3 = nn.Conv2d(in_channels=64, 
                              out_channels=64,
                              kernel_size=(5, 1), stride=(1,1),
                              padding=(0,0))
        self.bnAcc3 = nn.BatchNorm2d(64)
                              
        self.convAcc4 = nn.Conv2d(in_channels=64, 
                              out_channels=64,
                              kernel_size=(5,1), stride=(1, 1),
                              padding=(0, 0))
        self.maxPool2 = nn.MaxPool2d(kernel_size = (3,1))
                          
        self.bnAcc4 = nn.BatchNorm2d(64)

        self.lstmAcc1 = nn.LSTM(64*acc_features, hidden_size=128, num_layers=1, batch_first=True)
        self.lstmAcc2 = nn.LSTM(128, 128, num_layers=1, batch_first=True)


        self.convGyr1 = nn.Conv2d(in_channels=1, 
                              out_channels=64,
                              kernel_size=(5, 1), stride=(1,1),
                              padding=(0,0))
        self.bnGyr1 = nn.BatchNorm2d(64)
                              
        self.convGyr2 = nn.Conv2d(in_channels=64, 
                              out_channels=64,
                              kernel_size=(5,1), stride=(1, 1),
                              padding=(0, 0))
                              
        self.bnGyr2 = nn.BatchNorm2d(64)

        self.convGyr3 = nn.Conv2d(in_channels=64, 
                              out_channels=64,
                              kernel_size=(5, 1), stride=(1,1),
                              padding=(0,0))
        
        self.bnGyr3 = nn.BatchNorm2d(64)
        self.maxPool3 = nn.MaxPool2d(kernel_size = (3,1))
                      
        self.convGyr4 = nn.Conv2d(in_channels=64, 
                              out_channels=64,
                              kernel_size=(5,1), stride=(1, 1),
                              padding=(0, 0))
                              
        self.bnGyr4 = nn.BatchNorm2d(64)
        self.maxPool4 = nn.MaxPool2d(kernel_size = (3,1))

        self.lstmGyr1 = nn.LSTM(64*gyr_features, hidden_size=128, num_layers=1, batch_first=True)
        self.lstmGyr2 = nn.LSTM(128, 128, num_layers=1, batch_first=True)

        self.init_weight()

    def init_weight(self):
        init_layer(self.convAcc1)
        init_layer(self.convAcc2)
        init_layer(self.convAcc3)
        init_layer(self.convAcc4)
        init_layer(self.convGyr1)
        init_layer(self.convGyr2)
        init_layer(self.convGyr3)
        init_layer(self.convGyr4)


        init_bn(self.bnAcc1)
        init_bn(self.bnAcc2)
        init_bn(self.bnAcc3)
        init_bn(self.bnAcc4)
        init_bn(self.bnGyr1)
        init_bn(self.bnGyr2)
        init_bn(self.bnGyr3)
        init_bn(self.bnGyr4)

    def forward(self, inputAcc, inputGyr):

        x1 = self.convAcc1(inputAcc)
        x1 = self.bnAcc1(x1)
        x1 = self.convAcc2(x1)
        x1 = self.bnAcc2(x1)

        x1 = self.convAcc3(x1)
        x1 = self.bnAcc3(x1)
        x1 = self.convAcc4(x1)
        x1 = self.bnAcc4(x1)
        x1 = self.maxPool2(x1)

        x1 =  x1.reshape((x1.shape[0], x1.shape[2],-1)) 
        self.lstmAcc1.flatten_parameters()
        x1, _ = self.lstmAcc1(x1)
        self.lstmAcc2.flatten_parameters()
        x1, (h1,c1) = self.lstmAcc2(x1)

        x1 = torch.flatten(x1, start_dim=1)
        x2 = self.convGyr1(inputGyr)
        x2 = self.bnGyr1(x2)
        x2 = self.convGyr2(x2)
        x2 = self.bnGyr2(x2)

        x2 = self.convGyr3(x2)
        x2 = self.bnGyr3(x2)
        x2 = self.convGyr4(x2)
        x2 = self.bnGyr4(x2) 
        x2 = self.maxPool4(x2)

        x2 =  x2.reshape((x2.shape[0],x2.shape[2],-1)) 
        self.lstmGyr1.flatten_parameters()
        x2, _ = self.lstmGyr1(x2)
        self.lstmGyr2.flatten_parameters()
        x2 , (h2,c2) = self.lstmGyr2(x2)

        x2 = torch.flatten(x2, start_dim=1)

        x3 = torch.cat([x1, x2], 1)


        return x3

class DeepConvLSTM_Classifier(nn.Module):
    def __init__(
        self,
        model,
        input_dim,
        hidden_dim,
        filter_num,
        filter_size,
        enc_num_layers,
        enc_is_bidirectional,
        dropout,
        dropout_rnn,
        dropout_cls,
        activation,
        sa_div,
        num_class,
        train_mode,
        experiment,
        participant, **kwargs
    ):
        super(DeepConvLSTM_Classifier, self).__init__()

        self.experiment = f"train_{experiment}" if train_mode else experiment
        self.model = model
        self.hidden_dim = hidden_dim
        self.participant = participant
        print(paint(f"[STEP 3] Creating {self.model} HAR model ..."))

        self.fe = DeepConvLSTM_SplitV3(classes_num=num_class, acc_features=3, gyr_features = 3)

        self.dropout = nn.Dropout(dropout_cls)
        self.classifier = Classifier(hidden_dim, num_class)
        self.register_buffer(
            "centers", (torch.randn(num_class, self.hidden_dim).cuda())
        )

        # do not create log directories if we are only testing the models module
        if experiment != "test_models":
            if train_mode:
                makedir(self.path_checkpoints)
                makedir(self.path_logs)
            makedir(self.path_visuals)

    def forward(self, x):

        feature = self.fe(x[:,None,:,:3], x[:,None,:,3:])
        z = feature.div(
            torch.norm(feature, p=2, dim=1, keepdim=True).expand_as(feature)
        )
        logits = self.classifier(feature)
        return z, logits

    @property
    def path_checkpoints(self):
        return f"./models/{self.experiment}/checkpoints/{self.participant}"

    @property
    def path_logs(self):
        return f"./models/{self.experiment}/logs/{self.participant}"

    @property
    def path_visuals(self):
        return f"./models/{self.experiment}/visuals/{self.participant}"



class Attention(nn.Module):

    def __init__(self, dim, size):
        super().__init__()
        self.weight1 = torch.nn.Parameter(data = torch.Tensor(dim,size), requires_grad=True)
        self.b1 = torch.nn.Parameter(data = torch.Tensor(1, size), requires_grad=True)
        self.W = torch.nn.Parameter(data = torch.Tensor(size,1), requires_grad=True)


        self.init_weights()


    def init_weights(self):
        """Initialize a Linear or Convolutional layer. """
        nn.init.xavier_uniform_(self.weight1)
        nn.init.xavier_uniform_(self.W)
        nn.init.xavier_uniform_(self.b1)


    def forward(self, input, modality=False):
        x = torch.tensordot(input, self.weight1, dims=1)
        xx = x + self.b1

        if modality:
            xx_n = torch.linalg.norm(xx, dim=0, keepdims=True)
            xx = xx / xx_n
        xx = torch.tanh(xx)
        alpha = torch.tensordot(xx,self.W, dims=1)
        output = torch.sum(xx * alpha, dim = 1)
        return output, alpha

__factory = {
    "AttendDiscriminate": AttendDiscriminate,
    "AttendDiscriminate_MotionAudio": AttendDiscriminate_MotionAudio,
    'DeepConvLSTM_Classifier': DeepConvLSTM_Classifier,
    'Audio_CNN14': Audio_CNN14,
    "AttendDiscriminate_MotionAudio_CNN14": AttendDiscriminate_MotionAudio_CNN14,
    "AttendDiscriminate_MotionAudio_CNN14_Concatenate": AttendDiscriminate_MotionAudio_CNN14_Concatenate,
    "DeepConvLSTM_MotionAudio_CNN14_Concatenate": DeepConvLSTM_MotionAudio_CNN14_Concatenate,
    "DeepConvLSTM_MotionAudio_CNN14_Attention": DeepConvLSTM_MotionAudio_CNN14_Attention,
}


def create(model, config):
    if model not in __factory.keys():
        raise KeyError(f"[!] Unknown HAR model: {model}")
    return __factory[model](**config)