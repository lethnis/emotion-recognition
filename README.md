# Распознавание эмоций
Датасет был сгенерирован с использованием [модели Stable Diffusion](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0). Генерация изображений осуществлялась в google colab ([ноутбук](https://colab.research.google.com/drive/19e7sjE1R8dgxpzs7BU4_N6QO0UGVpAyp?usp=sharing)). Готовый датасет доступен [здесь](https://drive.google.com/drive/folders/1rg4Q_fYtbKaS6B2VZ6M0DcCgD9RLqqg5?usp=sharing). Для нахождения ключевых точек использовалась библиотека Mediapipe ([face landmarker model](https://developers.google.com/mediapipe/solutions/vision/face_landmarker#models)). Для классификации точек использовалась библиотека Scikit-learn.
# Результаты
<table>
  <tr>
    <td><img src="assets/smile-2072907_640.jpg" width=300></td>
    <td><img src="output/smile-2072907_640.jpg" width=300></td>
  </tr>
  <tr>
    <td><img src="assets/woman-1867127_640.jpg" width=300></td>
    <td><img src="output/woman-1867127_640.jpg" width=300></td>
  </tr>
  <tr>
    <td><img src="assets/surprised_2024-03-19 03_10_20.897183.jpg" width=300></td>
    <td><img src="output/surprised_2024-03-19 03_10_20.897183.jpg" width=300></td>
  </tr>
  <tr>
    <td><img src="assets/angry_2024-03-13 02_25_08.990517.jpg" width=300></td>
    <td><img src="output/angry_2024-03-13 02_25_08.990517.jpg" width=300></td>
  </tr>
  <tr>
    <td><img src="https://github.com/lethnis/emotion-recognition/assets/88483002/fdce6347-ae0a-4f01-8882-0b1c5de56a41" width=300></td>
    <td><img src="https://github.com/lethnis/emotion-recognition/assets/88483002/90e18f2b-d4c5-4bc0-8236-e0cba857ce57" width=300></td>
  </tr>
</table>

# Использование
1. Создать виртуальную среду с `python 3.11` и установить `requirements.txt`
2. Выполнить `python inference_model.py`. Можно дополнительно указать путь до папки или файла, которые нужно обработать. По умолчанию изображения и видео берутся из папки `assets`.
3. Результаты выполнения будут сохранены в папке `output`.
4. Для обработки с вебкамеры выполнить python `webcam.py`. Для выхода нажать `q`.

# Известные проблемы
Для обучения использовался небольшой набор искуственно сгенерированных данных. Модель имеет сложности с распознаванием эмоций с данных, отличающихся от тренировочных.
