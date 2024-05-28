#Create a Text Generation Model using ;GPT2, where input text should be given by the user

from transformers import GPT2LMHeadModel, GPT2Tokenizer

def generate_text(input_text, model_name, max_length, num_beams, no_repeat_ngram_size,
                  early_stopping):
    # Load pre-trained model and tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)

    # Set pad_token_id to eos_token_id because GPT2 does not have a pad_token_id
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Encode input text and create attention mask
    input_ids = tokenizer.encode(input_text, return_tensors='pt')
    # attention_mask = input_ids.ne(tokenizer.pad_token_id).long()

    # Generate text
    output = model.generate(
        input_ids,
        # attention_mask=attention_mask,
        max_length=max_length,
        num_beams=num_beams,
        no_repeat_ngram_size=no_repeat_ngram_size,
        early_stopping=early_stopping,
        pad_token_id=tokenizer.eos_token_id  # Set pad token ID to EOS token ID
    )

    # Decode and print the output
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text

# Get user input
input_text = input("Enter the initial text: ")
model_name = input("Enter the model name (default 'gpt2'): ") or 'gpt2'
max_length = int(input("Enter the maximum length of generated text (default 100): ") or 100)
num_beams = int(input("Enter the number of beams for beam search (default 5): ") or 5)
no_repeat_ngram_size = int(input("Enter the no repeat n-gram size (default 2): ") or 2)
early_stopping = input("Enable early stopping? (default 'yes'): ").lower() in ['yes', 'y', '']

# Generate and print the text
generated_text = generate_text(
    input_text,
    model_name=model_name,
    max_length=max_length,
    num_beams=num_beams,
    no_repeat_ngram_size=no_repeat_ngram_size,
    early_stopping=early_stopping
)
print("\nGenerated Text:\n")
print(generated_text)


#Create a Sentiment Analysis Model using Transformer Model “model_name = "distilbert-base-uncased-finetuned-sst-2-english"

from transformers import pipeline

# Explicitly specify the model name
model_name = "distilbert-base-uncased-finetuned-sst-2-english"

# Load the sentiment analysis pipeline with the specified model
sentiment_pipeline = pipeline("sentiment-analysis", model=model_name, tokenizer=model_name)

# Function to perform sentiment analysis
def analyze_sentiment(text):
    results = sentiment_pipeline(text)
    return results

# Example usage
if __name__ == "__main__":
    text = input("Enter the text for sentiment analysis: ")
    results = analyze_sentiment(text)
    for result in results:
        print(f"Label: {result['label']}, Score: {result['score']:.4f}")


