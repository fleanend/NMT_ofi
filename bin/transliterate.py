from src.models.set_up import set_up_model
from src.models.predict import transliterate

max_length_targ, max_length_inp, inp_lang, targ_lang, units, encoder, decoder = set_up_model()

while True:
    user_input = input("Pòula: ")
    print(transliterate(user_input, max_length_targ, max_length_inp, inp_lang, targ_lang, units, encoder, decoder))
    print()