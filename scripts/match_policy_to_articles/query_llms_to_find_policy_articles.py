from tqdm.auto import tqdm
import pandas as pd 

def make_sample_dataset(news_files):
    sample_set = []
    for n in tqdm(news_files):
        df = pd.read_json(n, lines=True)
        df_sample = (
            df.loc[lambda df: 
                df['article_publish_date'].notnull() & 
                (df['article_publish_date'] != None)
            ]
            .loc[lambda df: df['article_text'] != '']
            .sample(10_000)
        )
        t = df_sample[['article_url', 'article_publish_date', 'article_text']]
        sample_set.append(t)
    sample_df = pd.concat(sample_set)
    return sample_df


def make_prompts(sample_df, prompt_df_outfile='promt_df.csv'):
    prompt = """The following news article was published on:
            
        {publish_date}
    
        Does it directly cover a local city council meeting or other local government meeting occurring before the article was published?

        ```{news_article}```

        Answer with "Yes" or "No". Don't say anything else.
        """

    prompt_df = (
        sample_df
            .drop_duplicates('article_url')
            .set_index('article_url')
            .apply(lambda x: prompt.format(
                publish_date=x['article_publish_date'],
                news_article=x['article_text']
            ), axis=1)
            .sample(40_000)
            .drop_duplicates()
    )
    prompt_df.to_frame('prompt').to_csv(prompt_df_outfile)


def make_openai_prompt_df(prompt_df_infile='prompt_df.csv', openai_prompt_outfile='gpt-prompts.jsonl'):
    prompt_df = pd.read_csv(prompt_df_infile)
    gpt_prompt_df = prompt_df.apply(lambda x: {
         "custom_id": x['article_url'], 
         "method": "POST", 
         "url": "/v1/chat/completions", 
         "body": {
            "model": "gpt-4o-mini", 
            "messages": [{
                "role": "system", 
                "content": "You are a helpful journalist's assistant."
            },{
                "role": "user", "content": x['prompt']
            }],
         "max_tokens": 10
         }}, axis=1)
    with jsonlines.open(openai_prompt_outfile, 'w') as f:
        f.write_all(gpt_prompt_df.tolist())


"""
from openai import OpenAI
os.environ['OPENAI_API_KEY'] = open('/Users/spangher/.openai-my-project-key.txt').read().strip()
client = OpenAI()
batch_input_file = client.files.create(
    file=open("gpt-prompts.jsonl", "rb"),
    purpose="batch"
)
batch_input_file_id = batch_input_file.id




import glob
glob.glob('*.jsonl')
news_files_2

"""        
