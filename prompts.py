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
            Odpověz pouze následujícím formátem obsahujícím dvě položky:
            ### Analýza:
            Zde uveď důvod tvé odpovědi.
            ### Výsledek:
            ID autora dotazovaného textu.
            ''') + textwrap.dedent('''\
            Položka "Výsledek" musí obsahovat pouze číslo, které reprezentuje
            ID autora. Každý autor v seznamu autorů je jedinečný.
            Je zaručeno, že jeden z autorů v seznamu známých autorů
            napsal dotazovaný text. Důvod tvé odpovědi (Analýza) musí být
            v češtině, musí být formální a stručný a musí být v jednom odstavci
            bez jakéhokoliv formátování.
            ''').replace('\n', ' ')
    },
    {
        'role': 'user',
        'content': textwrap.dedent('''\
            Tvým úkolem je analyzovat poskytnuté texty a na základě stylistických charakteristik
            určit, kterému z autorů patří dotazovaný text. Zaměř se výhradně na styl psaní,
            ignoruj délku textu, funkční styl a téma textu.
            ''').replace('\n', ' ') + textwrap.dedent('''\n
            ### Dotazovaný text:
            {query_text}
            ### Texty potenciálních autorů:
            {example_text}
            ''')
    }
]

prompts_cz_gpt = [
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
            ignoruj délku textu, funkční styl a téma textu.
            ''').replace('\n', ' ') + textwrap.dedent('''\n
            ### Dotazovaný text:
            {query_text}
            ### Texty potenciálních autorů:
            {example_text}
            ''')
    }
]

prompts_cz_1shot_gpt = [
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
            ''').replace('\n', ' ') + textwrap.dedent('''\
            Příklad řešení:
            ### Dotazovaný text:
            Myslím, že jen netušíš kolik věcí se nikdy neprovalilo. Navíc jasně, že se nikdo nebude ptát proč a i kdyby nikdo mu neodpoví. Spousta státních úředníků skončila z ničeho nic. Mnoho z nich na vlastní žádost, která ovšem z pravidla není vlastní.  Za 20 let? To už nikdo z těch lidí na těch pozicích nebude a pravděpodobně bude v důchodu. To je win-win situace, koho zajímá co se stalo před 20 lety?
            ### Texty potenciálních autorů:
            Autor 95: Za předpokladu že se Argentinská cesta nevyjeví jako zázračně účinná, tak nejspíš nikdy. Je totiž mnohem efektivnější si raději peníze půjčit a zainvestovat je rovnou, než na danou investici šetřit. Důležitá je míra zadlužení a hlavně celková míra zadlužování se vůči růstu HDP, to totiž ovlivňuje výši úroku za který si půjčujeme.
            Autor 96: Halíře dělají talíře a s tvými návrhy souhlasím ač je všechno krom zdravotnictví docela nevýznamné. Ten problém proč tenhle příspěvek existuje je, že důchodců jsou 2M to jsou 2M hlasů pro toho kdo jim přidá, ikdyž to vůbec nedává smysl a v poměru ke mzdám mají nejvyšší důchod co kdy měli.
            Autor 97: A i přes takové případy existují lidé, kteří pití alkoholu omlouvají a dělají z něj normální věc. Ne, vážně to není normální ani v pořádku.  Vážím si všech, kdo jsou celoživotními abstinenty a myšlenku nepití, mezi lidi šíří
            Autor 98: Spíš u soudu se slzami v očích řekne, že od něj utekla, protože jí týral, a ještě půjde sedět. Bohužel čechyjská policie a soudy si u džina dali přání, aby dokázali přesně odhalit, kdo je vinen a kdo ne, ale džin byl zlý a překroutil ho, že musí jednat opačně, trestat nevinné a nechávat vinné. To a nebo to jsou prostě lotři.

            Odpověď:
            {
                "analysis": "Dotazovaný text a text autora s ID 96 sdílejí podobné stylografické prvky, jako je použití neformálního jazyka, krátkých vět a přímého vyjadřování. Oba texty obsahují hovorové výrazy a používají ironii a sarkasmus. Autor s ID 96 také často používá zkrácené věty a přímé otázky, což je patrné i v dotazovaném textu. Na rozdíl od textu autora s ID 95, který je formálnější a zaměřuje se na ekonomické termíny, a textu autora s ID 97, který je více moralistický a vážný, texty autora s ID 96 a dotazovaný text mají uvolněnější a kritičtější tón. Text autora s ID 98 je sice také kritický, ale používá více dramatických a literárních prvků, což se v dotazovaném textu nevyskytuje.",
                "answer": 96
            }
            
            Další možná odpověď:
            {
                "analysis": "Dotazovaný text vykazuje neformální styl psaní, což se projevuje například v použití hovorových výrazů a přímého vyjadřování. Tento styl je podobný textu autora s ID 96, který rovněž používá hovorový jazyk a krátké, úderné věty. Autor s ID 96 má tendenci formulovat své myšlenky přímo a bez zbytečné formality, což je patrné i v dotazovaném textu. Texty ostatních autorů, jako například autor s ID 95, jsou formálnější a odbornější, zatímco autor s ID 97 se vyjadřuje vážně a moralisticky. Autor s ID 98 používá dramatické a literární prvky, které v dotazovaném textu chybí. Tyto rozdíly v jazykových prvcích a stylu naznačují, že dotazovaný text nejvíce odpovídá stylu autora s ID 96.",
                "answer": 96
            }
            ''')
    },
    {
        'role': 'user',
        'content': textwrap.dedent('''\
            Tvým úkolem je analyzovat poskytnuté texty a na základě stylistických charakteristik
            určit, kterému z autorů patří dotazovaný text. Zaměř se výhradně na styl psaní,
            ignoruj délku textu, funkční styl a téma textu.
            ''').replace('\n', ' ') + textwrap.dedent('''\n
            ### Dotazovaný text:
            {query_text}
            ### Texty potenciálních autorů:
            {example_text}
            ''')
    }
]


prompts_cz_1shot = [
    {
        'role': 'system',
        'content': textwrap.dedent('''\
            Odpověz pouze následujícím formátem obsahujícím dvě položky:
            ### Analýza:
            Zde uveď důvod tvé odpovědi.
            ### Výsledek:
            ID autora dotazovaného textu.
            ''') + textwrap.dedent('''\
            Položka "Výsledek" musí obsahovat pouze číslo, které reprezentuje
            ID autora. Každý autor v seznamu autorů je jedinečný.
            Je zaručeno, že jeden z autorů v seznamu známých autorů
            napsal dotazovaný text. Důvod tvé odpovědi (Analýza) musí být
            v češtině, musí být formální a stručný a musí být v jednom odstavci
            bez jakéhokoliv formátování.
            ''').replace('\n', ' ') + textwrap.dedent('''\
            Příklad řešení:
            ### Dotazovaný text:
            Myslím, že jen netušíš kolik věcí se nikdy neprovalilo. Navíc jasně, že se nikdo nebude ptát proč a i kdyby nikdo mu neodpoví. Spousta státních úředníků skončila z ničeho nic. Mnoho z nich na vlastní žádost, která ovšem z pravidla není vlastní.  Za 20 let? To už nikdo z těch lidí na těch pozicích nebude a pravděpodobně bude v důchodu. To je win-win situace, koho zajímá co se stalo před 20 lety?
            ### Texty potenciálních autorů:
            {"95": "Za předpokladu že se Argentinská cesta nevyjeví jako zázračně účinná, tak nejspíš nikdy. Je totiž mnohem efektivnější si raději peníze půjčit a zainvestovat je rovnou, než na danou investici šetřit. Důležitá je míra zadlužení a hlavně celková míra zadlužování se vůči růstu HDP, to totiž ovlivňuje výši úroku za který si půjčujeme.", "96": "Halíře dělají talíře a s tvými návrhy souhlasím ač je všechno krom zdravotnictví docela nevýznamné. Ten problém proč tenhle příspěvek existuje je, že důchodců jsou 2M to jsou 2M hlasů pro toho kdo jim přidá, ikdyž to vůbec nedává smysl a v poměru ke mzdám mají nejvyšší důchod co kdy měli.", "97": "A i přes takové případy existují lidé, kteří pití alkoholu omlouvají a dělají z něj normální věc. Ne, vážně to není normální ani v pořádku.  Vážím si všech, kdo jsou celoživotními abstinenty a myšlenku nepití, mezi lidi šíří", "98": "Spíš u soudu se slzami v očích řekne, že od něj utekla, protože jí týral, a ještě půjde sedět. Bohužel čechyjská policie a soudy si u džina dali přání, aby dokázali přesně odhalit, kdo je vinen a kdo ne, ale džin byl zlý a překroutil ho, že musí jednat opačně, trestat nevinné a nechávat vinné. To a nebo to jsou prostě lotři."}
            Odpověď:
            ### Analýza:
            Dotazovaný text a text autora s ID 96 sdílejí podobné stylografické prvky, jako je použití neformálního jazyka, krátkých vět a přímého vyjadřování. Oba texty obsahují hovorové výrazy a používají ironii a sarkasmus. Autor s ID 96 také často používá zkrácené věty a přímé otázky, což je patrné i v dotazovaném textu. Na rozdíl od textu autora s ID 95, který je formálnější a zaměřuje se na ekonomické termíny, a textu autora s ID 97, který je více moralistický a vážný, texty autora s ID 96 a dotazovaný text mají uvolněnější a kritičtější tón. Text autora s ID 98 je sice také kritický, ale používá více dramatických a literárních prvků, což se v dotazovaném textu nevyskytuje.
            ### Výsledek:
            96
            Další možná odpověď:
            ### Analýza:
            Dotazovaný text vykazuje neformální styl psaní, což se projevuje například v použití hovorových výrazů a přímého vyjadřování. Tento styl je podobný textu autora s ID 96, který rovněž používá hovorový jazyk a krátké, úderné věty. Autor s ID 96 má tendenci formulovat své myšlenky přímo a bez zbytečné formality, což je patrné i v dotazovaném textu. Texty ostatních autorů, jako například autor s ID 95, jsou formálnější a odbornější, zatímco autor s ID 97 se vyjadřuje vážně a moralisticky. Autor s ID 98 používá dramatické a literární prvky, které v dotazovaném textu chybí. Tyto rozdíly v jazykových prvcích a stylu naznačují, že dotazovaný text nejvíce odpovídá stylu autora s ID 96.
            ### Výsledek:
            96
            ''')
    },
    {
        'role': 'user',
        'content': textwrap.dedent('''\
            Tvým úkolem je analyzovat poskytnuté texty a na základě stylistických charakteristik
            určit, kterému z autorů patří dotazovaný text. Zaměř se výhradně na styl psaní,
            ignoruj délku textu, funkční styl a téma textu.
            ''').replace('\n', ' ') + textwrap.dedent('''\n
            ### Dotazovaný text:
            {query_text}
            ### Texty potenciálních autorů:
            {example_text}
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
            o filmovou recenzi či publicistický článek. Pro analýzu použij pouze stylistické
            a jazykové rysy daných textů. Tvá odpověď musí mít maximálně 250 slov a musí být
            stručná, formální, v jednom odstavci a bez jakéhokoliv formátování. Svou odpověď
            začni analýzou dotazovaného textu, aniž bys jmenoval číslo daného autora.
            ''').replace('\n', ' ')
    },
    {
        'role': 'user',
        'content': textwrap.dedent('''\
            Dotazovaný text:
            {query_text}

            Texty potenciálních autorů:
            {example_text}

            Správná odpověď (autor dotazovaného textu): {correct_author}
            ''')
    }
]


prompts_cz_finetuning = [
    {
        'role': 'system',
        'content': textwrap.dedent('''\
            Odpověz pouze následujícím formátem obsahujícím dvě položky:
            ### Analýza:
            Zde uveď důvod tvé odpovědi.
            ### Výsledek:
            ID autora dotazovaného textu.
            ''') + textwrap.dedent('''\
            Položka "Výsledek" musí obsahovat pouze číslo, které reprezentuje
            ID autora. Každý autor v seznamu autorů je jedinečný.
            Je zaručeno, že jeden z autorů v seznamu známých autorů
            napsal dotazovaný text. Důvod tvé odpovědi (Analýza) musí být
            v češtině, musí být formální a stručný a musí být v jednom odstavci
            bez jakéhokoliv formátování.
            ''').replace('\n', ' ')
    },
    {
        'role': 'user',
        'content': textwrap.dedent('''\
            Tvým úkolem je analyzovat poskytnuté texty a na základě stylistických charakteristik
            určit, kterému z autorů patří dotazovaný text. Zaměř se výhradně na styl psaní,
            ignoruj délku textu, funkční styl a téma textu.
            ''').replace('\n', ' ') + textwrap.dedent('''\n
            ### Dotazovaný text:
            {query_text}
            ### Texty potenciálních autorů:
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

