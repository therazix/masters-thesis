import textwrap

prompts_en = {
    'system': textwrap.dedent('''\
        Respond with a JSON object including two key elements:
        {
          "analysis": Reasoning behind your answer.
          "answer": The query text's author ID.
        }'''),
    'user': textwrap.dedent('''\
        Given a set of texts with known authors and a query text, determine
        the author of the query text. Analyze the writing styles of the input
        texts, disregarding the differences in topic and content. Focus on
        linguistic features such as phrasal verbs, modal verbs, punctuation,
        rare words, affixes, quantities, humor, sarcasm, typographical errors,
        and misspellings."
        ''').replace('\n', ' ') + textwrap.dedent('''\n
        The input texts are delimited with triple backticks. ```
        
        Query text: {query}
        
        Texts from potential authors: {examples}
        
        ```
        ''')
}

prompts_cz = {
    'system': textwrap.dedent('''\
        Odpověz pouze JSON objektem obsahujícím dvě položky:
        {
        "analysis": "Zde uveď stručný důvod tvé odpovědi.",
        "answer": "ID autora dotazovaného textu."
        }''') + textwrap.dedent('''\
        Tvá odpověď musí být pouze platný JSON objekt, nic jiného. Všechny řetězce
        musí být ohraničeny dvojitými uvozovkami a mezi hodnotami musí
        být čárka. "answer" musí obsahovat pouze jedno číslo reprezentující
        ID autora. Nezapomeň správně ukončit JSON objekt složenou závorkou.
        Je zaručeno, že jeden z autorů v seznamu známých autorů
        napsal dotazovaný text. Důvod tvé odpovědi (analysis) musí být
        v češtině a měl by porovnávat jazykové rysy obou textů.
        ''').replace('\n', ' '),

    'user': textwrap.dedent('''\
        Na základě sady textů se známými autory a dotazovaného textu, urči
        autora dotazovaného textu. Analyzuj styl psaní vstupních textů,
        ignoruj rozdíly v tématu a obsahu. Zaměř se na jazykové rysy, jako
        jsou často opakovaná slova, interpunkce, vzácná slova,
        použití speciálních znaků, přípony, kvantifikátory, humor, sarkasmus,
        typografické chyby, překlepy a pohlaví autora.
        Použij metodu postupné racionalizace, kde postupně vysvětlíš, v čem
        jsou texty podobné a v čem se liší od ostatních.
        ''').replace('\n', ' ') + textwrap.dedent('''\n
        Dotazovaný text:
        {query}

        Texty potencionálních autorů:
        {examples}
        ''')
}
