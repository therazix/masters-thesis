import textwrap

MAX_TOKENS = 128000

mistral_prompts_en = {
    'system': textwrap.dedent('''\
        Respond with a JSON object including two key elements:
        "analysis": Reasoning behind your answer.
        "answer": The query text’s author ID.
        '''),
    'user': textwrap.dedent('''\
        Given a set of texts with known authors and a query text, determine
        the author of the query text. Analyze the writing styles
        of the input texts, disregarding the differences in topic and
        content. Focus on linguistic features such as phrasal verbs,
        modal verbs, punctuation, rare words, affixes, quantities,
        humor, sarcasm, typographical errors, and misspellings.
        The input texts are surrounded by triple backticks.
        ''').replace('\n', ' ') + textwrap.dedent('''\
        Query text: ``` {query} ```
        
        Texts from potential authors: ``` {examples} ```
        ''')
}
