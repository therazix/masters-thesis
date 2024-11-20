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

# prompts_cz = [
#     {
#         'role': 'system',
#         'content': textwrap.dedent('''\
#             Odpověz pouze JSON objektem obsahujícím dvě položky:
#             {
#             "analysis": "Zde uveď stručný důvod tvé odpovědi.",
#             "answer": "ID autora dotazovaného textu."
#             }
#             ''') + textwrap.dedent('''\
#             Položka "answer" musí obsahovat pouze číslo, které reprezentuje
#             ID autora. Je zaručeno, že jeden z autorů v seznamu známých autorů
#             napsal dotazovaný text. Důvod tvé odpovědi (analysis) musí být
#             v češtině a nesmí překročit více než 250 slov.
#             ''').replace('\n', ' ')
#     },
#     {
#         'role': 'user',
#         'content': textwrap.dedent('''\
#             Tvým úkolem je analyzovat poskytnuté texty a na základě stylistických charakteristik
#             určit, kterému z autorů patří dotazovaný text. Zaměř se výhradně na styl psaní,
#             ignoruj délku textu, funkční styl a téma textu. Sleduj zejména volbu slovní zásoby,
#             strukturu vět, preference zájmen a spojek, interpunkci, slovosled a frázovou strukturu.
#             Důležité jsou i vzorce ve skloňování a časování, míra formálnosti jazyka, používání
#             cizích či odborných termínů, humor, sarkasmus, specifické přípony a kvantifikátory,
#             použití speciálních znaků a typické překlepy nebo typografické chyby. Na základě
#             těchto stylistických prvků porovnej dotazovaný text s texty autorů. Hledej podobosti
#             a také rozdíly. Nakonec přiřaď text ke konkrétnímu autorovi. Napiš odpověď v maximálně
#             250 slovech, stručně, formálně, do jednoho odstavce a bez jakéhokoliv formátování.
#             ''').replace('\n', ' ') + textwrap.dedent('''\n
#             Dotazovaný text:
#             {query}
#
#             Texty potenciálních autorů:
#             {examples}
#             ''')
#     }
# ]

### BEST ONE? But with just default system prompt
# prompts_cz = [
#     {
#         'role': 'system',
#         'content': textwrap.dedent('''\
#             Odpověz pouze JSON objektem obsahujícím dvě položky:
#             {
#             "analysis": "Detaily tvé analýzy",
#             "answer": "ID autora dotazovaného textu."
#             }
#             ''') + textwrap.dedent('''\
#             Položka "answer" musí obsahovat pouze číslo, které reprezentuje
#             ID autora. Je zaručeno, že jeden z autorů v seznamu známých autorů
#             napsal dotazovaný text. Do položky "analysis" stručně, konkrétně
#             a bez jakéhokoliv formátování uveď hlavní hlavní důvody tvého rozhodnutí.
#             ''').replace('\n', ' ')
#     },
#     {
#         'role': 'user',
#         'content': textwrap.dedent('''\
#             Analyzuj poskytnuté texty a urči autora dotazovaného textu výhradně na základě unikátních stylistických charakteristik, bez ohledu na obsah a téma. Při analýze se zaměř na jemné odchylky v těchto prvcích: frekvenci a výběr konkrétních slov, strukturu vět včetně délky, složitosti a konzistentních vzorců slovosledu. Sleduj specifické preference v používání zájmen, spojek a částic, konzistenci ve vzorcích interpunkce, pravidelnosti v užívání cizích a odborných termínů a celkovou úroveň formálnosti. Zvláštní pozornost věnuj také tomu, jak často autor používá humor, ironii nebo sarkasmus, a jak pracuje s přídavnými jmény, kvantifikátory, příponami či neobvyklými znaky. Důležité jsou i běžné překlepy nebo typografické chyby specifické pro každého autora. Na základě těchto jemných stylistických prvků přiřaď text konkrétnímu autorovi.
#             ''').replace('\n', ' ') + textwrap.dedent('''\
#
#             Dotazovaný text:
#             {query}
#
#             Texty potencionálních autorů:
#             {examples}
#             ''')
#     }
# ]

prompts_cz = [
    {
        'role': 'system',
        'content': textwrap.dedent('''\
            Odpověz pouze JSON objektem obsahujícím dvě položky:
            {
            "analysis": "Zde uveď důvod tvé odpovědi.",
            "answer": "ID autora dotazovaného textu."
            }
            ''') + textwrap.dedent('''\
            Položka "answer" musí obsahovat pouze číslo, které reprezentuje
            ID autora. Každý autor v seznamu autorů je jedinečný.
            Je zaručeno, že jeden z autorů v seznamu známých autorů
            napsal dotazovaný text. Důvod tvé odpovědi (analysis) musí být
            v češtině, musí být formální a stručný a musí být v jednom odstavci
            bez jakéhokoliv formátování.
            ''').replace('\n', ' ')
    },
    {
        'role': 'user',
        'content': textwrap.dedent('''\
            Tvým úkolem je analyzovat poskytnuté texty a na základě stylistických charakteristik
            určit, kterému z autorů patří dotazovaný text. Zaměř se výhradně na styl psaní,
            ignoruj délku textu, funkční styl a téma textu. Sleduj zejména volbu slovní zásoby,
            strukturu vět, preference zájmen a spojek, interpunkci, vzorce ve skloňování
            a časování, míra formálnosti jazyka, používání cizích či odborných termínů,
            humor, sarkasmus, specifické přípony, kvantifikátory, použití speciálních znaků
            a typické překlepy nebo typografické chyby. Na základě
            těchto stylistických prvků porovnej dotazovaný text s texty autorů. Nakonec urči
            autora dotazovaného textu.
            ''').replace('\n', ' ') + textwrap.dedent('''\n
            Dotazovaný text:
            {query}

            Texty potenciálních autorů:
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
            v češtině a nesmí překročit více než 250 slov.
            ''').replace('\n', ' ')
    },
    {
        'role': 'user',
        'content': textwrap.dedent('''\
            Tvým úkolem je analyzovat poskytnuté texty a na základě stylistických charakteristik
            určit, kterému z autorů patří dotazovaný text. Zaměř se výhradně na styl psaní,
            ignoruj délku textu, funkční styl a téma textu. Sleduj zejména volbu slovní zásoby,
            strukturu vět, preference zájmen a spojek, interpunkci, slovosled a frázovou strukturu.
            Důležité jsou i vzorce ve skloňování a časování, míra formálnosti jazyka, používání
            cizích či odborných termínů, humor, sarkasmus, specifické přípony a kvantifikátory,
            použití speciálních znaků a typické překlepy nebo typografické chyby. Na základě
            těchto stylistických prvků porovnej dotazovaný text s texty autorů. Hledej podobosti
            a také rozdíly. Nakonec přiřaď text ke konkrétnímu autorovi. Napiš odpověď v maximálně
            250 slovech, stručně, formálně, do jednoho odstavce a bez jakéhokoliv formátování.
            ''').replace('\n', ' ') + textwrap.dedent('''\n
            Dotazovaný text:
            "Myslím, že jen netušíš kolik věcí se nikdy neprovalilo. Navíc jasně, že se nikdo nebude ptát proč a i kdyby nikdo mu neodpoví. Spousta státních úředníků skončila z ničeho nic. Mnoho z nich na vlastní žádost, která ovšem z pravidla není vlastní.  Za 20 let? To už nikdo z těch lidí na těch pozicích nebude a pravděpodobně bude v důchodu. To je win-win situace, koho zajímá co se stalo před 20 lety?"        
            
            Texty potenciálních autorů:
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
            "analysis": "Oba texty používají neformální jazyk a hovorové výrazy, což se projevuje v přímém a nekomplikovaném vyjadřování. V obou textech je patrná preference pro používání zájmen a spojek, které zajišťují plynulost a propojení myšlenek. Struktura vět je v obou případech spíše jednoduchá a přímočará, což přispívá k jasnosti sdělení. Interpunkce je používána standardně, bez nadměrného využívání speciálních znaků nebo neobvyklých typografických prvků. Oba texty také vykazují určitou míru sarkasmu a kritického pohledu na danou problematiku, což je patrné v ironickém tónu a skeptickém přístupu k popisovaným situacím. V dotazovaném textu i textu autora 96 je přítomna tendence k zobecňování a používání kvantifikátorů, které posilují argumentaci. Celkově je jazyk obou textů neformální, s důrazem na přímé vyjádření názorů a postojů. Tyto stylistické prvky naznačují, že dotazovaný text by mohl být dílem autora 96.",
            "answer": 96
            }
            ''')
    },
    {
        'role': 'user',
        'content': textwrap.dedent('''\
            Dotazovaný text:
            {query}
    
            Texty potenciálních autorů:
            {examples}
            ''')
    }
]

prompts_finetuning = [
    {
        'role': 'system',
        'content': textwrap.dedent('''\
            Analyzuj dotazovaný text a texty uvedených autorů a následně urči, proč styl psaní
            dotazovaného textu odpovídá uvedenému autorovi. Urči také, čím se liší od ostatních
            autorů. Seznam neobsahuje žádné známé spisovatele, proto autory nazývej pouze dle
            jejich čísel (ID) a nepoužívej žádná jména. Zaměř se výhradně na styl psaní, ignoruj
            délku textu, funkční styl a téma textu. Ignoruj například to, jestli se jedná
            o filmovou recenzi či publicistický článek. Pro analýzu použij následující stylistické
            charakteristiky: volba slovní zásoby, struktura vět, preference zájmen a spojek,
            interpunkce, slovosled, frázová struktura, vzorce ve skloňování a časování, míra
            formálnosti jazyka, používání cizích či odborných termínů, humor, sarkasmus,
            specifické přípony a kvantifikátory, použití speciálních znaků a typické překlepy
            nebo typografické chyby. Tvá odpověď musí mít maximálně 250 slov a musí být stručná,
            formální, v jednom odstavci a bez jakéhokoliv formátování.
            ''').replace('\n', ' ')
    },
    {
        'role': 'user',
        'content': textwrap.dedent('''\
            Dotazovaný text:
            "{query}"

            Texty potenciálních autorů:
            {examples}

            Správná odpověď (autor dotazovaného textu): {correct_author}
            ''')
    }
]

prompts_cz_finetuning = [
    {
        'role': 'system',
        'content': textwrap.dedent('''\
            Jsi odborník na rozpoznání autorství. Tvým úkolem je analyzovat
            zadaný text a určit, který autor z uvedeného seznamu jej napsal.
            Seznam autorů a jejich příslušné ukázkové texty jsou unikátní
            pro tuto konverzaci. Pomocí stylistických a jazykových rysů
            porovnej zadaný text s ukázkovými texty a identifikuj správného
            autora. Tvá odpověď musí být v následujícím formátu:
            ''').replace('\n', ' ') + textwrap.dedent('''\n
            ### Analýza:
            Podrobný popis stylistických rysů zadaného textu a jejich shody s vybraným autorem.
            
            ### Výsledek:
            Číslo odpovídající správnému autorovi.
            ''')
    },
    {
        "role": "user",
        "content": textwrap.dedent('''\
            ### Zadaný text:
            "{query_text}"
            
            ### Seznam autorů:
            {example_text}
            ''')
    },
    {
        "role": "assistant",
        "content": textwrap.dedent('''\
            ### Analýza:
            {response}
            
            ### Výsledek:
            {label}
            ''')
    }
]

prompts_cz_inference = [
    {
        'role': 'system',
        'content': textwrap.dedent('''\
            Jsi odborník na rozpoznání autorství. Tvým úkolem je analyzovat
            zadaný text a určit, který autor z uvedeného seznamu jej napsal.
            Seznam autorů a jejich příslušné ukázkové texty jsou unikátní
            pro tuto konverzaci. Pomocí stylistických a jazykových rysů
            porovnej zadaný text s ukázkovými texty a identifikuj správného
            autora. Tvá odpověď musí být v následujícím formátu:
            ''').replace('\n', ' ') + textwrap.dedent('''\n
            ### Analýza:
            Podrobný popis stylistických rysů zadaného textu a jejich shody s vybraným autorem.

            ### Výsledek:
            Číslo odpovídající správnému autorovi.
            ''')
    },
    {
        "role": "user",
        "content": textwrap.dedent('''\
            ### Zadaný text:
            "{query_text}"

            ### Seznam autorů:
            {example_text}
            ''')
    }
]
