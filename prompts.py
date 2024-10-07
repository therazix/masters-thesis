import textwrap

MAX_TOKENS = 128000

mistral_prompts_en = {
    'system': textwrap.dedent('''\
        Respond with a JSON object including two key elements:
        {
        "analysis": "Reasoning behind your answer.",
        "answer": "The query text’s author ID."
        }
        Make sure your response is a valid JSON object, all strings
        must be surrounded by double quotes and there must be a comma
        between the key-value pairs.
        '''),
    'user': textwrap.dedent('''\
        Given a set of texts with known authors and
        a query text written in Czech, determine
        the author of the query text. Analyze the writing styles
        of the input texts, disregarding the differences in topic and
        content. Focus on linguistic features such as gender,
        modal verbs, punctuation, rare words, affixes, quantities,
        humor, sarcasm, typographical errors, and misspellings.
        The input texts are surrounded by triple backticks.
        ''').replace('\n', ' ') + textwrap.dedent('''\n
        Query text: ``` {query} ```
        
        Texts from potential authors: ``` {examples} ```
        ''')
}

mistral_prompts_cz = {
    'system': textwrap.dedent('''\
        Odpověz JSON objektem obsahujícím dvě položky:
        "analysis": "Důvod tvé odpovědi. Odpověz jasně a stručně. Odpověď piš pouze v češtině.",
        "answer": "ID autora dotazovaného textu."
        '''),
    'user': textwrap.dedent('''\
        Na základě sady textů se známými autory a dotazovaného textu,
        urči autora dotazovaného textu. Analyzuj styl psaní vstupních
        textů, ignoruj rozdíly v tématu a obsahu. Zaměř se na jazykové
        rysy, jako jsou modální slovesa, interpunkce, často opakovaná slova, vzácná slova,
        použití speciálních znaků, přípony, kvantifikátory, humor, sarkasmus,
        typografické chyby, překlepy a pohlaví autora. Vstupní texty jsou ohraničeny
        trojitými zpětnými apostrofy.
        ''').replace('\n', ' ') + textwrap.dedent('''\n
        Dotazovaný text: ``` {query} ```

        Texty potencionálních autorů: ``` {examples} ```
        ''')
}
