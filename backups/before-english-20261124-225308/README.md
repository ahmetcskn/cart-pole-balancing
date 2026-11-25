# Stick Balance

C++ ile yazılmış araba–sarkaç simülasyonu. PyTorch ile eğitilen model, arabayı x ekseninde hareket ettirerek çubuğu dengede tutmaya çalışır. Tork kullanılmaz.

Fizik ve görüntü C++ tarafında, eğitim Python tarafındadır. Bağlantı pybind11 ile kurulur. Eğitimi GLFW penceresinde canlı izleyebilirsiniz.

## Gereksinimler

C++17, CMake, Python 3, PyTorch, NumPy, pybind11 ve GLFW/OpenGL/GLUT.

## Kullanım

Komutları proje klasöründe çalıştırın. C++ bağlantısı otomatik derlenir.

```bash
# Sıfırdan eğitim başlat
python3 train.py

# Kayıtlı modeli izle
python3 train.py --play

# İsmini verdiğin modeli izle (yalnızca dosya adı yaz)
python3 train.py --play --model model_v2.pt

# Pencere açmadan eğitim yap
python3 train.py --headless

# Eğitim başlatmadan fizik ve bağlantıyı kontrol et
python3 train.py --self-test
```

Durdurmak için `Ctrl+C` kullanın veya pencereyi kapatın.

## Model kayıtları

Varsayılan eğitim modeli `rl_runs/balance/policy.pt` dosyasına kaydedilir. Her yeni eğitim sıfırdan başlar ve bu dosyayı günceller. Eski modeli korumak istiyorsanız önce farklı bir adla kopyalayın.

`--play` yalnızca modeli çalıştırır, eğitim yapmaz.

## Dosyalar

- `src/cart_pole.cpp`: fizik motoru
- `src/viewer.cpp`: GLFW görüntüsü
- `src/python_bridge.cpp`: pybind11 bağlantısı
- `train.py`: sinir ağı ve PPO eğitimi
