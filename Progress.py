import time

class Progress():
    def __init__(self, iterable, metrics = None):
        self.iterable = iterable
        self.metrics = metrics

    def __iter__(self):
        print(f'-.--ms/iteration  |  0.00%  |  ETA --:--:--', end='\r')

        iterable = self.iterable
        metrics = self.metrics

        if hasattr(iterable, '__len__'):
            size = len(iterable)
        elif hasattr(iterable, 'shape'):
            size = iterable.shape[0]
        elif hasattr(iterable, '__length_hint__'):
            size = iterable.__length_hint__()
        else:
            raise Exception('Failed to compute size of iterable object')

        target_ns = 1.0 * 1e9

        n = 0
        prev_n = 0
        n_iter = 1
        start = time.clock_gettime_ns(time.CLOCK_MONOTONIC)

        try:
            for item in iterable:
                yield item

                n += 1

                if n - prev_n >= n_iter:
                    cur = time.clock_gettime_ns(time.CLOCK_MONOTONIC)

                    total_time = cur - start
                    per_itr = total_time / n

                    n_iter = int(target_ns / per_itr + 1)

                    eta_s = (size - n) * per_itr * 1e-9
                    eta_h = int(eta_s // 3600)
                    eta_s -= eta_h * 3600
                    eta_m = int(eta_s // 60)
                    eta_s -= eta_m * 60
                    eta_s = int(eta_s + 0.5)
                    s = ''
                    if metrics is not None:
                        s = '  |  '.join(['{}: {}'.format(key, val) for key, val in metrics.items()]) + '  |  '
                    print(s + '{:7.2f}ms/iteration  |  {:.2f}%  |  ETA {:02d}:{:02d}:{:02d}            '.format(
                          per_itr * 1e-6, 100.0 * n / size, eta_h, eta_m, eta_s), end='\r')
                    prev_n = n

        finally:
            cur = time.clock_gettime_ns(time.CLOCK_MONOTONIC)
            total_time = cur - start
            per_itr = total_time / (n + 1e-8)
            elapsed_s = (cur - start) * 1e-9
            elapsed_h = int(elapsed_s // 3600)
            elapsed_s -= elapsed_h * 3600
            elapsed_m = int(elapsed_s // 60)
            elapsed_s -= elapsed_m * 60
            elapsed_s = int(elapsed_s + 0.5)

            s = ''
            if metrics is not None:
                s = '  |  '.join(['{}: {}'.format(key, val) for key, val in metrics.items()]) + '  |  '

            print(s + '{:7.2f}ms/iteration  |  100.00%  |  Elapsed {:02d}:{:02d}:{:02d}'.format(
                  per_itr * 1e-6, elapsed_h, elapsed_m, elapsed_s))
