from LexRank.LexRank import (LexRank, CNNDailyMailCorpus)


def main(text: str) -> None:
    lr = LexRank()
    summary = lr.summarize(text)

if __name__ == "__main__":
    text = """
    Klimatförändringar påverkar världen på många sätt. Extrema väderhändelser som värmeböljor, torka och översvämningar blir allt vanligare.
    Forskare varnar för att den globala uppvärmningen kan leda till stora förändringar i ekosystem och livsmedelsförsörjning.
    För att begränsa temperaturökningen krävs kraftiga minskningar av utsläppen av växthusgaser.
    Många länder investerar nu i förnybar energi, som sol- och vindkraft.
    Samtidigt behövs internationellt samarbete för att hantera de långsiktiga effekterna av klimatförändringarna.
    """
    main(text)