# 01_env_check.py  (T+0, 약 1분)  ── 환경 점검: 파이썬/CPU/라이브러리/GPU/인터넷
# 사용: env_check()   (Jupyter 에서는 %load snippets/01_env_check.py 로 불러와 실행)
import sys, os, platform, importlib

def env_check(extra=()):
    print('python', sys.version.split()[0], '|', platform.platform(), '| CPU 스레드', os.cpu_count())
    libs = ['numpy', 'pandas', 'sklearn', 'scipy', 'lightgbm', 'xgboost', 'catboost',
            'torch', 'torchvision', 'transformers', 'cv2', 'PIL', 'ultralytics', *extra]
    for m in libs:
        try:
            mod = importlib.import_module(m); print(f'  {m:13s} {getattr(mod, "__version__", "ok")}')
        except Exception:
            print(f'  {m:13s} -  (없음: 필요하면 !pip install, 허용 목록·허락 확인)')
    try:
        import torch
        print('torch  cuda:', torch.cuda.is_available(), '| 스레드:', torch.get_num_threads())
    except Exception:
        pass
    try:
        import urllib.request
        urllib.request.urlopen('https://pypi.org', timeout=5); print('인터넷: 가능')
    except Exception:
        print('인터넷: 불가(다운로드가 필요한 방법은 쓰지 않는다)')

if __name__ == '__main__':
    env_check()
