from utils import *

keywords = KeywordsLoader('./bright_keywords')
mask_creator = MaskCreator('bigram_model.pkl', keywords.nouns, keywords.nouns_actors)

sentence = 'У Миши было 3 три мячика. Два из них он отдал Даше. Сколько мячиков осталось у Миши?'
print(convert(mask_creator, keywords, sentence, 2, 0, True, False))