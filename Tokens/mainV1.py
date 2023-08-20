import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# Sample text data
texts = [
    "Once upon a time",
    "In a land far away",
    "There lived a",
    "Brave knight named",
    "Sir Lancelot",
    "He was known throughout the kingdom",
    "For his courage and chivalry",
    "One day a mysterious",
    "castle stood tall against the horizon",
    "shrouded in mist and legends",
    "its walls whispered stories of ancient battles",
    "as the wind carried tales of old",
    "knights donned in shining armor",
    "set out on quests to prove their worth",
    "dragons roared and treasure gleamed",
    "underneath the starlit skies",
    "magic flowed like a river",
    "enchanted forests guarded secrets untold",
    "and every stone had a story to share",
    "the jester's laughter echoed through the halls",
    "as kings and queens ruled with grace",
    "and the realm was alive with dreams",
    "of heroes and destinies intertwined",
    "for in this world of imagination",
    "words held the power to create worlds anew",
    "beneath the silver moon's gentle glow",
    "whispers of enchantment filled the air",
    "a quest for the lost city began",
    "where forgotten treasures waited to be found",
    "across the rolling hills and meadows",
    "adventurers embarked on epic journeys",
    "seeking the answers to ancient riddles",
    "as the sun painted the sky with hues of gold",
    "villages came to life with bustling markets",
    "and storytellers captivated young hearts",
    "legends were woven into tapestries",
    "unveiling tales of bravery and sacrifice",
    "the sea's waves sang a soothing lullaby",
    "while sailors navigated by the stars",
    "the air was filled with the scent of adventure",
    "as explorers mapped uncharted lands",
    "and friendships were forged through trials",
    "fireside tales kept the spirit of magic alive",
    "wizards conjured spells in ancient libraries",
    "while alchemists brewed potions of wonder",
    "bards strummed melodies of old",
    "inspiring love, hope, and laughter",
    "and when darkness threatened to consume",
    "heroes emerged from the shadows",
    "to defend the realm against all odds",
    "and so the cycle of tales continued",
    "for each word held a universe of possibilities",
    "where dreams and reality intertwined",
    "painting a canvas of stories untamed",
    "and reminding us that imagination knows no bounds",
    "underneath the twinkling canopy of stars",
    "the universe whispered secrets to those who listened",
    "galaxies danced in cosmic ballets",
    "as stardust paved the way for new beginnings",
    "lost civilizations left their echoes in the sands of time",
    "as archaeologists unearthed fragments of history",
    "ancient ruins stood as silent witnesses",
    "to the mysteries of civilizations long past",
    "the symphony of nature played on",
    "with melodies of wind, rain, and rustling leaves",
    "mountains stood tall, guardians of the earth",
    "while rivers wove stories through valleys",
    "wildflowers painted landscapes with color",
    "and the sun kissed the world with warmth",
    "in the heart of bustling cities",
    "streets teemed with life, stories, and dreams",
    "artists captured fleeting moments on canvas",
    "while poets penned verses that echoed through time",
    "bridges connected people, places, and stories",
    "as cultures intermingled in a tapestry of diversity",
    "laughter echoed through parks and squares",
    "as families and friends created memories",
    "the passage of time was marked by seasons",
    "each bringing its own chapter to life's story",
    "from spring's rebirth to winter's hushed beauty",
    "life flowed in an eternal dance of change",
    "and so the world turned, a living chronicle",
    "where every soul added a line to the narrative",
    "a never-ending tale of existence and wonder",
    "each moment a paragraph in the book of life",
    "as the pages turned, stories continued to unfold",
    "and the legacy of humanity was etched in time",
    "for stories are the threads that weave us together",
    "connecting hearts, transcending boundaries",
    "a testament to our shared journey through life",
    "and a celebration of the human spirit's boundless creativity",
    "so let the stories flow like rivers of inspiration",
    "and let us be storytellers, crafting our destiny"
]


# Tokenize and preprocess the data
tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)
total_words = len(tokenizer.word_index) + 1

sequences = tokenizer.texts_to_sequences(texts)
max_sequence_length = max([len(seq) for seq in sequences])
input_sequences = pad_sequences(sequences, maxlen=max_sequence_length, padding='pre')

# Create input sequences and labels
input_sequences = np.array(input_sequences)
labels = np.zeros_like(input_sequences)
labels[:, :-1] = input_sequences[:, 1:]

# Build the model
embedding_dim = 50
lstm_units = 100

model = Sequential([
    Embedding(total_words, embedding_dim, input_length=max_sequence_length),
    LSTM(lstm_units, return_sequences=True),
    Dense(total_words, activation='softmax')
])

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')

# Train the model
num_epochs = 100
model.fit(input_sequences, labels, epochs=num_epochs, verbose=1)

# Generate text
seed_text = "once "
num_words_to_generate = 10

for _ in range(num_words_to_generate):
    encoded = tokenizer.texts_to_sequences([seed_text])[0]
    encoded = pad_sequences([encoded], maxlen=max_sequence_length, padding='pre')
    predicted = model.predict(encoded, verbose=0)
    predicted_word_index = np.argmax(predicted[0][-1])
    next_word = tokenizer.index_word[predicted_word_index + 1]  # Fix KeyError by adding 1
    seed_text += " " + next_word

model.save('LanguageModelV1.keras')

print(seed_text)
