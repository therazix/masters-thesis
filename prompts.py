import textwrap


prompts_en = [
    {
        'role': 'system',
        'content': textwrap.dedent('''\
            Respond with a JSON object including two key elements:
            {
              "analysis": Reasoning behind your answer.
              "answer": The query text's author ID.
            }
            ''')
    },
    {
        'role': 'user',
        'content': textwrap.dedent('''\
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
]

prompts_cz = [
    {
        'role': 'system',
        'content': textwrap.dedent('''\
            Odpověz pouze JSON objektem obsahujícím dvě položky:
            {
            "analysis": "Zde uveď stručný důvod tvé odpovědi.",
            "answer": "ID autora dotazovaného textu."
            }
            ''') + textwrap.dedent('''\
            Položka "answer" musí obsahovat pouze číslo, které reprezentuje
            ID autora. Je zaručeno, že jeden z autorů v seznamu známých autorů
            napsal dotazovaný text. Důvod tvé odpovědi (analysis) musí být
            v češtině a nesmí překročit více než 200 slov.
            ''').replace('\n', ' ')
    },
    {
        'role': 'user',
        'content': textwrap.dedent('''\
            Tvým úkolem je analyzovat poskytnuté texty a na základě stylistických charakteristik
            určit, kterému z autorů patří dotazovaný text. Zaměř se výhradně na styl psaní,
            ignoruj téma a obsah textu. Sleduj zejména volbu slovní zásoby, strukturu vět,
            preference zájmen a spojek, interpunkci, slovosled a frázovou strukturu. Důležité jsou
            i vzorce ve skloňování a časování, míra formálnosti jazyka, používání cizích
            či odborných termínů, humor, sarkasmus, specifické přípony a kvantifikátory, použití
            speciálních znaků a typické překlepy nebo typografické chyby. Na základě těchto
            stylistických prvků přiřaď text ke konkrétnímu autorovi.
            ''').replace('\n', ' ') + textwrap.dedent('''\n
            Dotazovaný text:
            {query}
    
            Texty potencionálních autorů:
            {examples}
            ''')
    }
]


prompts_cz_1shot = [
    {
        'role': 'system',
        'content': textwrap.dedent('''\
            Odpověz pouze JSON objektem obsahujícím dvě položky:
            {
            "analysis": "Zde uveď stručný důvod tvé odpovědi.",
            "answer": "ID autora dotazovaného textu."
            }
            ''') + textwrap.dedent('''\
            Položka "answer" musí obsahovat pouze číslo, které reprezentuje
            ID autora. Je zaručeno, že jeden z autorů v seznamu známých autorů
            napsal dotazovaný text. Důvod tvé odpovědi (analysis) musí být
            v češtině a nesmí překročit více než 200 slov.
            ''').replace('\n', ' ')
    },
    {
        'role': 'user',
        'content': textwrap.dedent('''\
            Tvým úkolem je analyzovat poskytnuté texty a na základě stylistických charakteristik
            určit, kterému z autorů patří dotazovaný text. Zaměř se výhradně na styl psaní,
            ignoruj téma a obsah textu. Sleduj zejména volbu slovní zásoby, strukturu vět,
            preference zájmen a spojek, interpunkci, slovosled a frázovou strukturu. Důležité jsou
            i vzorce ve skloňování a časování, míra formálnosti jazyka, používání cizích
            či odborných termínů, humor, sarkasmus, specifické přípony a kvantifikátory, použití
            speciálních znaků a typické překlepy nebo typografické chyby. Na základě těchto
            stylistických prvků přiřaď text ke konkrétnímu autorovi.
            ''').replace('\n', ' ') + textwrap.dedent('''\n
            Dotazovaný text:
            "Myslím, že jen netušíš kolik věcí se nikdy neprovalilo. Navíc jasně, že se nikdo nebude ptát proč a i kdyby nikdo mu neodpoví. Spousta státních úředníků skončila z ničeho nic. Mnoho z nich na vlastní žádost, která ovšem z pravidla není vlastní.  Za 20 let? To už nikdo z těch lidí na těch pozicích nebude a pravděpodobně bude v důchodu. To je win-win situace, koho zajímá co se stalo před 20 lety?"        
            
            Texty potencionálních autorů:
            {
            "95": "Za předpokladu že se Argentinská cesta nevyjeví jako zázračně účinná, tak nejspíš nikdy. Je totiž mnohem efektivnější si raději peníze půjčit a zainvestovat je rovnou, než na danou investici šetřit. Důležitá je míra zadlužení a hlavně celková míra zadlužování se vůči růstu HDP, to totiž ovlivňuje výši úroku za který si půjčujeme.",
            "96": "Halíře dělají talíře a s tvými návrhy souhlasím ač je všechno krom zdravotnictví docela nevýznamné. Ten problém proč tenhle příspěvek existuje je, že důchodců jsou 2M to jsou 2M hlasů pro toho kdo jim přidá, ikdyž to vůbec nedává smysl a v poměru ke mzdám mají nejvyšší důchod co kdy měli.",
            "97": "A i přes takové případy existují lidé, kteří pití alkoholu omlouvají a dělají z něj normální věc. Ne, vážně to není normální ani v pořádku.  Vážím si všech, kdo jsou celoživotními abstinenty a myšlenku nepití, mezi lidi šíří",
            "98": "Spíš u soudu se slzami v očích řekne, že od něj utekla, protože jí týral, a ještě půjde sedět. Bohužel čechyjská policie a soudy si u džina dali přání, aby dokázali přesně odhalit, kdo je vinen a kdo ne, ale džin byl zlý a překroutil ho, že musí jednat opačně, trestat nevinné a nechávat vinné. To a nebo to jsou prostě lotři."
            }
            ''')
    },
    {
        'role': 'assistant',
        'content': textwrap.dedent('''\
            {
            "analysis": "Styl textu odpovídá autorovi 96, především díky podobné míře formálnosti, specifické struktuře vět a volbě slov. Autor 96 využívá neformální jazyk, kombinaci krátkých a složitějších souvětí a frekventovaně používá obecná zájmena, což vidíme i v dotazovaném textu („nikdo“, „spousta“, „mnoho“). Oba texty mají spíše cynický, pragmatický tón a užívají obecně přijímané fráze (např. „kdo jim přidá“ či „pravděpodobně bude v důchodu“), což podtrhuje běžný styl vyjadřování. Významnou paralelou je také preference pro přirozený slovosled a využívání nenápadné ironie, aniž by text explicitně přecházel do sarkasmu. Naopak ostatní autoři vykazují výrazně odlišné stylistické rysy: autor 95 používá odbornější terminologii a složitější syntax; autor 97 zaujímá morální pozici s jednoduchou strukturou vět; autor 98 používá sarkastické, expresivní obraty a netradiční spojení.",
            "answer": 96
            }
            ''')
    },
    {
        'role': 'user',
        'content': textwrap.dedent('''\
            Dotazovaný text:
            {query}
    
            Texty potencionálních autorů:
            {examples}
            ''')
    }
]
