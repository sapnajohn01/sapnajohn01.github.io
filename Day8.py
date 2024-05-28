#Create a Language translation model where user can select options for 1. Hindi 2. German and accordingly output will be printed

from transformers import MarianMTModel, MarianTokenizer
# Function to perform translation
def translate(text, choice):

    match choice:
        case 1:
            #target_language = "Hindi"
            model_name = "Helsinki-NLP/opus-mt-en-hi"
        case 2:
            #target_language = "German"
            model_name = "Helsinki-NLP/opus-mt-en-de"
        case _:
            raise ValueError("Invalid choice. Please select 1 for Hindi or 2 for German.")

    # Load model and tokenizer
    model = MarianMTModel.from_pretrained(model_name)
    tokenizer = MarianTokenizer.from_pretrained(model_name)

    # Prepare text for the model
    inputs = tokenizer(text, return_tensors="pt", padding=True)

    # Generate translation
    translated = model.generate(**inputs)
    translated_text = tokenizer.batch_decode(translated, skip_special_tokens=True)[0]

    return translated_text

if __name__ == "__main__":
    text = input("Enter the text to be translated: ")
    choice = int(input("Select your Choice(Number): 1:  Hindi, 2: German): "))

    try:
        translated_text = translate(text, choice)
        print(f"Translated text: {translated_text}")
    except ValueError as e:
        print(e)

