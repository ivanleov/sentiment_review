import re


def preprocess_reviews(input_text):
    # Добавляем перенос строки перед любой цифрой, за которой следует точка с запятой
    processed_text = re.sub(r'(\d+;)', r'\n\1', input_text)
    # Убираем перенос строки в начале, если он появился
    return processed_text.strip()


def save_reviews_to_file(input_text, output_file="reviews.txt"):
    if not input_text:
        print("Ошибка: входной текст пустой")
        return

    try:
        # Предварительно обрабатываем текст
        processed_text = preprocess_reviews(input_text)

        # Записываем в файл
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(processed_text)

        print(f"Отзывы успешно записаны в файл '{output_file}'")

    except Exception as e:
        print(f"Произошла ошибка при записи в файл: {e}")


if __name__ == "__main__":
    print("Введите текст отзывов (все отзывы в одну строку):")
    input_text = input()
    save_reviews_to_file(input_text)