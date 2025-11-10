import json
import csv
import matplotlib.pyplot as plt
import tqdm

API_KEY=''

from openai import OpenAI
client = OpenAI(api_key=API_KEY)

def summarize_reasons(rs, nwords=5):
    shorts = []
    for r in rs:
        res = summarize(r, nwords)
        shorts.append(res)
    return shorts
        
        
def summarize(text, nwords=5):
    response = client.responses.create(
        model="gpt-4o",
        input=f"summarize reason in the following text in {nwords} words: \n\n {text} "
    )
    return response.output_text 

if __name__ == "__main__":
    f = open("cvsc-reasons.json")
    res = json.load(f)
    responses = res['graph_data']['edges']
    with open('gpt4o-mini-country_vs_country-reasons.csv', 'w', newline='') as file:
        writer = csv.writer(file,delimiter='|')
        j = 0
        for k, v in tqdm.tqdm(responses.items()):
            j = j+1
            countA = v['option_A']['description'].split()[0]
            countryA = " ".join(v['option_A']['description'].split()[3:-5])
            countB = v['option_B']['description'].split()[0]
            countryB = " ".join(v['option_B']['description'].split()[3:-5])
            if countryA == countryB:
                continue
            aux_data = v['aux_data']
            if 'is_pseudolabel' in aux_data and aux_data['is_pseudolabel'] == True:
                continue
            probA = float(v['probability_A'])
            if countA == 'You' or countB == 'You':
                continue
            print(f"shortening {j}")
            for i in range(1):
                short_reason = summarize(aux_data['original_reasoning'][i], 5)
                writer.writerow([int(countA), countryA, int(countB), countryB, aux_data['original_parsed'][i], short_reason, aux_data['original_reasoning'][i]])
            for i in range(1):
                short_reason = summarize(aux_data['original_reasoning'][i], 5)  
                writer.writerow([int(countB), countryB, int(countA), countryA, aux_data['flipped_parsed'][i], short_reason, aux_data['flipped_reasoning'][i]])        

    
