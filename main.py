import json  # для сохранения токенов в формате json
import logging
import os.path  # для сохранения токенов в формате json
import io  # для сохранения токенов в формате json
from PyQt5 import uic  # интерфейс
from PyQt5.QtWidgets import QApplication  # интерфейс
import keras  # модели
from keras.src.legacy.preprocessing.text import tokenizer_from_json  # для сохр токенизатора
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, Dropout, Bidirectional, SimpleRNN,GlobalMaxPooling1D
from tensorflow.keras import utils
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import regularizers
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import confusion_matrix
import seaborn as sns


import pandas as pd  # ексель выборки
import numpy as np
from matplotlib import pyplot as plt  # график точности
plt.rcParams.update({
    'font.size': 20,  # Base font size
    'axes.titlesize': 24,  # Title font size
    'axes.labelsize': 24,  # Axis label size
    'xtick.labelsize': 20,  # X-axis tick labels
    'ytick.labelsize': 20,  # Y-axis tick labels
    'legend.fontsize': 24,  # Legend font size
    'figure.titlesize': 20  # Figure title size
})
from tensorflow.python.keras.models import save_model
from nltk.corpus import stopwords  # добавить импорт для стоп-слов
import nltk
# Add these imports at the top of main.py
from PyQt5.QtWidgets import QFileDialog, QMessageBox
from PyQt5.QtCore import QTimer


print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

Form, Window = uic.loadUiType("1.ui")  # интерфейс
app = QApplication([])
window = Window()
form = Form()
form.setupUi(window)
window.show()
num_words = 50000  # колво отзывов
max_review_len = 500  # макс длина отзыва из базы

nltk.download('stopwords')
russian_stopwords = set(stopwords.words('russian'))

def remove_stopwords(review):
    if isinstance(review, str):
        tokens = review.split()
        filtered_tokens = [token for token in tokens if token.lower() not in russian_stopwords]
        return ' '.join(filtered_tokens)
    return review

if os.path.exists('model.keras'):  # если модель нейронки существует подключать её
    model = keras.models.load_model('model.keras')
    with open('tokenizer.json') as f:  # импорт ТОКЕНАЙЗЕРА
        data = json.load(f)
        tokenizer = tokenizer_from_json(data)


else:  # иначе создавай модель, обучай её, выводи график обучения и сохраняй ее

    tokenizer = Tokenizer(num_words=num_words)  # токенизатор - присваиваем словам номера по частоте

    train = pd.read_excel('viborka/top_reviews.xlsx',  # наша выборка проверочная (парсинг яндекс карт)
                          header=0,
                          names=['Class', 'Review'])
    test = pd.read_excel('viborka/test_reviews.xlsx',  # наша выборка тестовая (отзывы от GPT)
                         header=0,
                         names=['Class', 'Review'])




    reviews = train['Review'].apply(remove_stopwords)  # обработать обучающие отзывы, удаляя стоп-слова
    test_reviews = test['Review'].apply(remove_stopwords)  # обработать тестовые отзывы, удаляя стоп-слова
    tokenizer.fit_on_texts(reviews)

    y_train = to_categorical(train['Class'] - 1, num_classes=5)
    y_test = to_categorical(test['Class'] - 1, num_classes=5)

    sequences = tokenizer.texts_to_sequences(reviews)
    test_sequences = tokenizer.texts_to_sequences(test_reviews)

    tokenizer_json = tokenizer.to_json()  # СОХРАНЕНИЕ ТОКЕНАЙЗЕРА
    with io.open('tokenizer.json', 'w', encoding='utf-8') as f:
        f.write(json.dumps(tokenizer_json, ensure_ascii=False))

    x_train = pad_sequences(sequences, maxlen=max_review_len)
    x_test = pad_sequences(test_sequences, maxlen=max_review_len)



    model = models.Sequential([
        layers.Embedding(input_dim=50000, output_dim=128),
        layers.GlobalMaxPooling1D(),
        layers.Dense(128,activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(64,activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(32,activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(5, activation='softmax')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )


    # Создаём колбэк для вывода точности по каждому классу:
    class PerClassAccuracyCallback(tf.keras.callbacks.Callback):
        def __init__(self):
            super().__init__()
            self.class_accuracies = {i: [] for i in range(5)}  # Store accuracy for each class
            self.epochs = []

        def on_epoch_end(self, epoch, logs=None):
            preds = self.model.predict(x_test)
            y_true = np.argmax(y_test, axis=1)
            y_pred = np.argmax(preds, axis=1)

            correct_by_class = [0] * 5
            total_by_class = [0] * 5
            for i in range(len(y_true)):
                cls_true = y_true[i]
                total_by_class[cls_true] += 1
                if y_pred[i] == cls_true:
                    correct_by_class[cls_true] += 1

            print(f"\nТочность предсказания {epoch + 1} эпохи для каждого класса: ")
            for i in range(5):
                if total_by_class[i] > 0:
                    acc = correct_by_class[i] / total_by_class[i]
                else:
                    acc = 0
                print(f"  Класс {i + 1}: {acc:.4f}")
                self.class_accuracies[i].append(acc)

            self.epochs.append(epoch + 1)

        def plot_class_accuracies(self):
            plt.figure(figsize=(10, 6))
            class_labels = ["1 звезда", "2 звезды", "3 звезды", "4 звезды", "5 звезд"]

            for i in range(5):
                plt.plot(self.epochs, self.class_accuracies[i], marker='o', label=f'Класс {class_labels[i]}')

            plt.title('Точность модели для каждого класса')
            plt.xlabel('Эпоха')
            plt.ylabel('Точность')
            plt.legend(loc='lower right')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.show()



    per_class_callback = PerClassAccuracyCallback()

    history = model.fit(
        x_train,
        y_train,
        epochs=13,
        batch_size=16,
        validation_data=(x_test, y_test),
        callbacks=[per_class_callback]
    )

    # Plot accuracy per class
    per_class_callback.plot_class_accuracies()


    def plot_confusion_matrix(y_true, y_pred, class_names):
        # Создаем матрицу ошибок
        cm = confusion_matrix(y_true, y_pred)
        total_samples = len(y_true)

        # Вывод информации о матрице для отладки
        print(f"Размер матрицы ошибок: {cm.shape}")
        print(f"Всего образцов: {total_samples}")
        print("Матрица ошибок (абсолютные значения):")
        print(cm)

        # 1. Матрица абсолютных значений
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        plt.title('Матрица ошибок (абсолютные значения)')
        plt.ylabel('Истинный класс')
        plt.xlabel('Предсказанный класс')
        plt.tight_layout()
        plt.savefig('confusion_matrix_absolute.png')  # Сохраняем как файл
        plt.show(block=True)  # Принудительно отображаем

        # 2. Нормализованная по строкам матрица (процент для каждого истинного класса)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', xticklabels=class_names,
                    yticklabels=class_names)
        plt.title('Нормализованная матрица ошибок (по классам)')
        plt.ylabel('Истинный класс')
        plt.xlabel('Предсказанный класс')
        plt.tight_layout()
        plt.savefig('confusion_matrix_normalized.png')  # Сохраняем как файл
        plt.show(block=True)  # Принудительно отображаем


    # Генерация матрицы ошибок после обучения
    print("\nСоздание матрицы ошибок...")
    preds = model.predict(x_test)
    y_true = np.argmax(y_test, axis=1)
    y_pred = np.argmax(preds, axis=1)
    class_labels = ["1 звезда", "2 звезды", "3 звезды", "4 звезды", "5 звезд"]
    plot_confusion_matrix(y_true, y_pred, class_labels)
    # Your existing code for plotting overall accuracy and loss
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Параметры модели')
    plt.xlabel('Эпоха')
    plt.legend(['обучающая выборка точность', 'тестовая выборка точность', 'обучающая выборка потери',
                'тестовая выборка потери'], loc='upper left')
    plt.xlim(1, 12)
    plt.show()


    def plot_top_words_per_class(top_words_dict, num_words=10):
        """
        Plot top informative words for each class with no overlapping.

        Parameters:
        - top_words_dict: Dictionary with class labels as keys and lists of (word, score) tuples as values
        - num_words: Number of top words to display per class
        """
        class_labels = ["1 звезда", "2 звезды", "3 звезды", "4 звезды", "5 звезд"]
        num_classes = len(class_labels)

        # Create figure with subplots - one per class
        fig, axes = plt.subplots(1, num_classes, figsize=(24, 8))
        fig.suptitle('Наиболее информативные слова для каждого класса', fontsize=24)

        for i, (label, ax) in enumerate(zip(class_labels, axes)):
            # Get top words for this class
            if i in top_words_dict:
                words = top_words_dict[i][:num_words]
                words_text, scores = zip(*words)

                # Horizontal bar chart
                y_pos = np.arange(len(words_text))
                ax.barh(y_pos, scores, align='center')
                ax.set_yticks(y_pos)
                ax.set_yticklabels(words_text)
                ax.invert_yaxis()  # Labels read top-to-bottom
                ax.set_title(label)

                # Remove spines
                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_visible(False)

        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust spacing to prevent overlap with title
        plt.savefig('top_words_per_class.png', dpi=300, bbox_inches='tight')
        plt.show()


    # Python
    def extract_important_words(model, tokenizer, class_count=5):
        """
        Extract most important words for each class from a trained model.

        Returns a dictionary with class indices as keys and lists of (word, importance) tuples as values.
        """
        word_importance = {}

        # Get weights from the last dense layer (shape: (n, num_classes))
        weights = model.layers[-1].get_weights()[0]

        # Get word index from tokenizer
        word_index = tokenizer.word_index

        # For each class
        for class_idx in range(class_count):
            word_scores = {}

            # Calculate importance for each word
            for word, idx in word_index.items():
                # Ensure index is within weights matrix bounds
                if (idx - 1) < weights.shape[0]:
                    word_scores[word] = abs(weights[idx - 1, class_idx])

            # Sort words by importance
            sorted_words = sorted(word_scores.items(), key=lambda x: x[1], reverse=True)
            word_importance[class_idx] = sorted_words

        return word_importance


    # Call the function
    class_labels = ["1 звезда", "2 звезды", "3 звезды", "4 звезды", "5 звезд"]
    top_words = extract_important_words(model, tokenizer)
    plot_top_words_per_class(top_words)


    # Print and visualize the most informative words
    plt.figure(figsize=(15, 12))
    for class_idx in range(5):
        plt.subplot(3, 2, class_idx + 1)

        words = [word for word, _ in top_words[class_idx]]
        scores = [score for _, score in top_words[class_idx]]

        plt.barh(words, scores)
        plt.title(f"Наиболее информативные слова для класса '{class_labels[class_idx]}'")
        plt.xlabel('Влияние на предсказание')
        plt.tight_layout()

    plt.savefig('most_informative_words.png')
    plt.show()

    # Print in text form as well
    print("Наиболее информативные слова для каждого класса:")
    for class_idx in range(5):
        print(f"\nКласс '{class_labels[class_idx]}':")
        for word, score in top_words[class_idx]:
            print(f"  {word}: {score:.4f}")
    model.save('model.keras')  # сохраняем полученную модель


def getlabel():  # бери написанный в окошке текст
    return form.textEdit.toPlainText()


def clickb1():  # по нажатии на кнопку прогоняй текст через нейронку
    text = (str(getlabel()))  # переводи текст на англ
    text = remove_stopwords(text).lower()
    sequence = tokenizer.texts_to_sequences([text])  # токенизируй
    data = pad_sequences(sequence, maxlen=max_review_len)
    result = model.predict(data)  # получай результат 0 или 1

    class_labels = ["1 звезда", " 2 звезды", "3 звезды", "4 звезды", "5 звезд"]  # обновите, если у вас есть метки классов
    predicted_class = np.argmax(result)  # получить индекс класса с нам большей вероятностью
    text = f"Результат оценки: {class_labels[predicted_class]}"

    form.label_3.setText(text)  # помести результат  в окно


form.pushButton.clicked.connect(clickb1)  # метод по нажатии на кнопку



# Add these functions after the clickb1 function

# Updated browse_file function
def browse_file():
    file_path, _ = QFileDialog.getOpenFileName(window,
                                               "Выбрать CSV файл с отзывами",
                                               "",
                                               "CSV файлы (*.csv);;Все файлы (*)")
    if file_path:
        form.lineEdit_filepath.setText(file_path)
        # Try to load column names to help user
        try:
            delimiter = get_delimiter()
            df = pd.read_csv(file_path, delimiter=delimiter, nrows=0)
            columns = df.columns.tolist()
            if columns:
                form.lineEdit_column.setPlaceholderText(f"Доступные столбцы: {', '.join(columns)}")
        except Exception:
            pass


# Helper function to get the selected delimiter
def get_delimiter():
    delimiter_idx = form.comboBox_delimiter.currentIndex()
    if delimiter_idx == 0:
        return ','
    elif delimiter_idx == 1:
        return ';'
    elif delimiter_idx == 2:
        return '\t'
    return ','


# Updated analyze_file function
def analyze_file():
    file_path = form.lineEdit_filepath.text()
    if not file_path:
        QMessageBox.warning(window, "Предупреждение", "Пожалуйста, выберите файл для анализа!")
        return

    try:
        # Get column name or index for reviews
        column = form.lineEdit_column.text()
        delimiter = get_delimiter()

        # Read CSV file using pandas
        df = pd.read_csv(file_path, delimiter=delimiter)

        # Check if column exists
        if column.isdigit():
            # User specified column by index
            column_idx = int(column)
            if column_idx >= len(df.columns):
                raise ValueError(
                    f"Индекс столбца {column_idx} вне диапазона. Доступные столбцы: 0-{len(df.columns) - 1}")
            reviews = df.iloc[:, column_idx].astype(str).tolist()
        else:
            # User specified column by name
            if column not in df.columns:
                raise ValueError(f"Столбец '{column}' не найден. Доступные столбцы: {', '.join(df.columns)}")
            reviews = df[column].astype(str).tolist()

        # Remove NaN values and empty strings
        reviews = [review for review in reviews if review and review.strip() and review.lower() != 'nan']

        # Initialize counters
        total_reviews = len(reviews)
        star_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}  # Using 0-based indexes
        results = []

        # Initialize progress bar
        form.progressBar.setMaximum(total_reviews)
        form.progressBar.setValue(0)

        # Process each review
        for i, review in enumerate(reviews):
            review = review.strip()
            if not review:
                continue

            # Preprocess and predict
            processed_text = remove_stopwords(review).lower()
            sequence = tokenizer.texts_to_sequences([processed_text])
            data = pad_sequences(sequence, maxlen=max_review_len)
            prediction = model.predict(data, verbose=0)
            predicted_class = np.argmax(prediction)

            # Update counters
            star_counts[predicted_class] += 1

            # Store result
            class_labels = ["1 звезда", "2 звезды", "3 звезды", "4 звезды", "5 звезд"]
            results.append(
                f"{i + 1}. {review[:50]}{'...' if len(review) > 50 else ''} - {class_labels[predicted_class]}")

            # Update progress bar
            form.progressBar.setValue(i + 1)

            # Process UI events to keep the interface responsive
            app.processEvents()

        # Calculate percentages
        percentages = {}
        for star in range(5):
            percentages[star] = (star_counts[star] / total_reviews) * 100 if total_reviews > 0 else 0

        # Update statistics display
        form.label_total.setText(f"Всего отзывов: {total_reviews}")
        form.label_star1.setText(f"1 звезда: {star_counts[0]} ({percentages[0]:.1f}%)")
        form.label_star2.setText(f"2 звезды: {star_counts[1]} ({percentages[1]:.1f}%)")
        form.label_star3.setText(f"3 звезды: {star_counts[2]} ({percentages[2]:.1f}%)")
        form.label_star4.setText(f"4 звезды: {star_counts[3]} ({percentages[3]:.1f}%)")
        form.label_star5.setText(f"5 звезд: {star_counts[4]} ({percentages[4]:.1f}%)")

        # Display detailed results
        form.textEdit_results.setText("\n".join(results))

        # Generate and display a pie chart of the results
        plt.figure(figsize=(8, 8))
        plt.pie([star_counts[i] for i in range(5)],
                labels=["1 звезда", "2 звезды", "3 звезды", "4 звезды", "5 звезд"],
                autopct='%1.1f%%',
                startangle=90)
        plt.axis('equal')
        plt.title('Распределение оценок')
        plt.savefig('ratings_distribution.png')
        plt.close()

        reviews_data = []
        ratings_data = []

        for result in results:
            parts = result.split(' - ')
            if len(parts) >= 2:
                review_part = parts[0]
                rating = parts[1]

                # Extract actual review text, handling potential formatting issues
                if '.' in review_part:
                    number_and_review = review_part.split('. ', 1)
                    if len(number_and_review) > 1:
                        review_text = number_and_review[1]
                    else:
                        review_text = number_and_review[0]
                else:
                    review_text = review_part

                reviews_data.append(review_text)
                ratings_data.append(rating)

        # Create DataFrame with the extracted data
        # Save results to CSV
        result_df = pd.DataFrame({
            'Review': [r.split(' - ')[0].split('. ', 1)[1] for r in results],
            'Rating': [r.split(' - ')[1] for r in results]
        })

        result_df.to_csv('sentiment_analysis_results.csv', index=False)

        QMessageBox.information(window, "Анализ завершен",
                                f"Анализ {total_reviews} отзывов завершен успешно.\n"
                                f"Результаты сохранены в файлах:\n"
                                f"- ratings_distribution.png (диаграмма)\n"
                                f"- sentiment_analysis_results.csv (таблица)")

    except Exception as e:
        QMessageBox.critical(window, "Ошибка", f"Произошла ошибка при анализе файла:\n{str(e)}")


# Connect the new buttons (add these lines before app.exec_())
form.pushButton_browse.clicked.connect(browse_file)
form.pushButton_analyze.clicked.connect(analyze_file)

app.exec_()  # покидай интерфейс
