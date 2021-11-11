import numpy as np
import librosa
import os


n_fft = 1024
n_frame = 40
hope_size = 512
lpc_dim = 16

input_wav = 'c135ab5f.wav'


class lpc:
  def __init__(self, toeplitz_elements):
    self.toeplitz_elements = toeplitz_elements

  def solve(self):
    solutions, extra_ele = self.solve_size_one()
    final_solutions, _ = self.solve_recursively(solutions, extra_ele)
    return np.delete(final_solutions, 0)

  def solve_size_one(self):
    solutions = np.array([1.0, - self.toeplitz_elements[1] / self.toeplitz_elements[0]])
    extra_element = self.toeplitz_elements[0] + self.toeplitz_elements[1] * solutions[1]
    return solutions, extra_element

  def solve_recursively(self, initial_solutions, initial_extra_ele):
    extra_element = initial_extra_ele
    solutions = initial_solutions
    for k in range(1, lpc_dim):
      lambda_value = self._calculate_lambda(k, solutions, extra_element)
      extended_solution = self._extend_solution(solutions)
      r_extended_solution = extended_solution[::-1]

      solutions = extended_solution + lambda_value * r_extended_solution
      extra_element = self._calculate_extra_element(extra_element, lambda_value)
    return solutions, extra_element

  def _extend_solution(self, previous_solution):
    return np.hstack((previous_solution, np.array([0.0])))

  def _calculate_extra_element(self, previous_extra_ele, lambda_value):
    return (1.0 - lambda_value**2) * previous_extra_ele

  def _calculate_lambda(self, k, solutions, extra_element):
    tmp_value = 0.0
    for j in range(0, k + 1):
      tmp_value += (- solutions[j] * self.toeplitz_elements[k + 1 - j])
    return tmp_value / extra_element


def autocorr(x):
  x = np.atleast_1d(x)
  if len(x.shape) != 1:
    raise ValueError("invalid dimensions for 1D autocorrelation function")
  n = next_pow_two(len(x))
  f = np.fft.fft(x - np.mean(x), n=2 * n)
  acf = np.fft.ifft(f * np.conjugate(f))[: len(x)].real
  acf /= acf[0]
  return acf


def next_pow_two(n):
  i = 1
  while i < n:
    i = i << 1
  return i
        

if __name__ == '__main__':
  indices = np.tile(np.arange(0, n_fft), (n_frame, 1)) + np.tile(np.arange(0, n_frame * hope_size, hope_size), (n_fft, 1)).T

  y, sr = librosa.load(input_wav, 16000)
  y = librosa.util.fix_length(y, 20480 + 512)

  frames = y[indices.astype(np.int32, copy=False)]
  frames *= np.hamming(1024)

  result = []

  for idx, frame in enumerate(frames):
    if not frame.any():
      solutions = np.zeros(lpc_dim)
    else:
      acorr = autocorr(frame)
      ld = lpc(acorr)
      solutions = ld.solve()
    result.append(solutions)

  np.savetxt(input_wav.replace('wav', 'txt'), result, fmt='%f', delimiter='\n')
