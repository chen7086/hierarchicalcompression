# demo.py
import warnings
import logging

warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)

from hc_compressor import hierarchical_compress

# ==== MODIFY THESE THREE VALUES AS NEEDED ====
QUESTION = "What is the place of birth of Carrie Watson Fleming's husband?"

CONTENT = """Since Peter's death, Fleming and her sister Kate have controlled Ian Fleming Publications. Passage 4: Anne Fleming (writer) Anne Fleming (born 25 April 1964) is a Canadian fiction writer. Born in Toronto, Ontario, Fleming attended the University of Waterloo, enrolling in a geography program then moving to English studies. In 1991, she moved to British Columbia. She teaches at the University of British Columbia Okanagan campus in Kelowna. She formerly taught at the Victoria School of Writing. Her fiction has been published in magazines and anthologies, including Toronto Life magazine, The Journey Prize Stories, and The New Quarterly, where it won a National Magazine Award. Her first book, Pool-Hopping and Other Stories, was a finalist at the 1999 Governor General's Awards; it was also a contender for the Ethel Wilson Fiction Prize and the Danuta Gleed Award. Her second book is the novel, Anomaly (Raincoast Books 2005). Aside from her literary endeavors, Fleming has hosted a radio program, played defense for the Vancouver Voyagers women's hockey team, and also plays the ukulele. She has a partner and a child. Fleming's great-grandfather was the mayor of Toronto, and Toronto figures prominently in her writing. In 2013 she served alongside Amber Dawn and Vivek Shraya on the jury of the Dayne Ogilvie Prize, a literary award for LGBT writers in Canada, selecting C. E. Gatchalian as that year's winner. Bibliography Pool-Hopping and Other Stories, 1998 (ISBN 1-896095-18-6) Anomaly, 2005 (ISBN 1-55192-831-0) Gay Dwarves of America, 2012 (ISBN 1897141467) poemw, 2016 (ISBN 1-897141-76-9) The Goat, 2017 (ISBN 1-55498-917-5) See also List of University of Waterloo people Passage 5: Carrie Watson Fleming Caroline Margaret Watson Fleming (1844–1931) was the wife of former Governor of West Virginia Aretas B. Fleming and served as that state's First Lady from 1890 to 1893. Biography Fleming was born on April 29, 1844, at Fairmont, West Virginia, a daughter of Matilda Lamb and James Otis Watson, early coal operators in that region. She attended Mount de Chantal Visitation Academy at Wheeling, West Virginia. In September 1865, she married the attorney for oil and gas magnate Johnson N. Camden: Aretas B. Fleming. They had one child, a daughter. Her husband, Aretas Fleming, served as Governor of West Virginia from 1890-1893. After a shortened three year term as first lady, due to the controversial 1888 election, the Flemings returned to Fairmont, where she played a prominent role in Fairmont's cultural, civic, and religious activities. In 1916, she signed a memorial from the West Virginia Association Opposed to Woman Suffrage that was sent to the legislature as they considered an amendment to the state constitution to expand the franchise to include women. Fleming died at Fairmont on July 19, 1931, at the age of 87. She is buried next to her husband in the Woodlawn Cemetery. Passage 6: Stan Marks Stan Marks is an Australian writer and journalist. He is the husband of Holocaust survivor Eva Marks. Life Born in London, Marks moved to Australia aged two. He became a reporter on rural daily papers and then on the State's evening The Herald (Melbourne), reporting and acting as a critic in the Melbourne and Sydney offices. He worked in London, Canada and in New York City for Australian journals. Back in Australia, Stan Marks became Public Relations and Publicity Supervisor for the Australian Broadcasting Commission, looking after television, radio and concerts, including publicity for Isaac Stern, Yehudi Menuhin, Igor Stravinsky, Daniel Barenboim, Maureen Forrester and international orchestras for Radio Australia and the magazine TVTimes. Later he became Public Relations and Publicity Manager for the Australian Tourist Commission, writing articles for newspapers and journals at home and abroad. Marks was also the editor of the Centre News magazine of the Jewish Holocaust Museum and Research Centre for over 16 years."""

TARGET_TOKENS = 300
# =============================================

def main():
    print("\n=== Original Prompt ===")
    print(f"{CONTENT}")

    print("\n=== Question ===")
    print(f"{QUESTION}")
    
    print("\n=== Target Token ===")
    print(f"{TARGET_TOKENS}")
    # run hierarchical compression
    compressed = hierarchical_compress(QUESTION, CONTENT, TARGET_TOKENS)
    print("\n=== Compressed Output ===")
    print(compressed)

if __name__ == "__main__":
    main()
