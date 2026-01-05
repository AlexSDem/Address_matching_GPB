!apt-get update
!apt-get install -y osmium-tool

# 0. ставим основные либы
!pip install osmnx pandas geopandas

# 1. Обновляем pip и ставим Cython (нужен для сборки)
!pip install --upgrade pip
!pip install Cython

# 2. Ставим pygeos (самая проблемная зависимость)
!pip install pygeos

# 3. Теперь ставим pyrosm
!pip install pyrosm

# 4. ставим geocoder
!pip install reverse_geocoder
!pip install nlpaug
!pip install natasha
!pip install rapidfuzz
