import anthropic
import pandas as pd
from anthropic.types.beta.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.beta.messages.batch_create_params import Request

class SynGen:

    client: anthropic.Anthropic
    model: str
    real_data: pd.DataFrame

    SYSTEM_PROMPT = """
    You are a machine learning academic working in the field of computational 
    social science. You are working on the low-resource domain of Filipino 
    tweets where data annotation for your problem is scarce. Your research area 
    is hate speech detection where you are trying to classify tweets based on 
    two labels: 0 for not hate speech and 1 for hate speech. You are trying 
    to improve this low-resource domain through data augmentation.
    """

    PROMPT = """
    Generate exactly one sentence of a Filipino tweet about the elections that 
    can be used to train a model for hate speech detection. Given is a label 
    in the form of integer where 0 means the predicted text is NOT hate speech 
    and 1 if the predicted text to be hate speech. For a label, a prediction 
    is generated where you generatee exactly one sentence following that label.

    The sample should mimic real-life examples, including the linguistic features
    such as punctuation, grammar and other linguistic fingerprints.
    Return only the prediction without comments and without additional text.
    
    <example>
    {INPUT}
    </example>

    Label: {TARGET_LABEL}
    Prediction: 
    """

    def __init__(self, real_data, model='claude-3-5-sonnet-20241022', api_key=None) -> None:
        self.client = anthropic.Anthropic(
            api_key=api_key
        )
        self.real_data = real_data
        self.model = model

    def generate(self, n=2):
        labels = self.real_data['label'].unique()
        n_per_labels = n // labels.shape[0]


        requests = []

        for label in labels:
            input_str = self.__stringify(self.real_data)
            
            for i in range(n_per_labels):
                req = self.__generate_batch_request(
                    input_str, label, f'synthetic-{label}-{i}')
                requests.append(req)

        self.message_batch = self.client.beta.messages.batches.create(requests=requests)        
                

    def test_get_examples(self):
        return self.__stringify(self.real_data)
    

    def get_token_count(self, df: pd.DataFrame):
        for _, x in df.iterrows():
            print(x['text'])

        return ''

    def get_results(self):

        tweet_dict = {
            'text': [],
            'label': [],
            'n_tokens': []
        }

        try:

            for result in self.client.beta.messages.batches.results(
                self.message_batch.id,
            ):
                tweet = self.__extract_results(result)
                tweet_dict['text'].append(tweet['content'])
                tweet_dict['label'].append(tweet['label'])
                tweet_dict['n_tokens'].append(tweet['tokens'])
        except anthropic.AnthropicError:
            return 'Generating still in progress...'

        return pd.DataFrame(tweet_dict)

    def __generate_batch_request(self, input_data, target_label, custom_id):

        return Request(
            custom_id=custom_id,
            params=MessageCreateParamsNonStreaming(
                model=self.model,
                temperature=1,
                max_tokens=1024,
                system=self.SYSTEM_PROMPT,
                messages=[{
                    'role': 'user',
                    'content': self.PROMPT.format(INPUT=input_data, TARGET_LABEL=target_label),
                }]
            )

        )

    def __stringify(self, data: pd.DataFrame):
        _data = data.groupby('label').sample(25)
        input_str = ''

        for _, x in _data.iterrows():
            input_str += f'Label: {x.iloc[1]}\n'
            input_str += f'Prediction: {x.iloc[0]}\n'

        return input_str
    
    def __extract_results(self, result):
        return {
            'content': result.result.message.content[0].text,
            'tokens': result.result.message.usage.output_tokens,
            'label': result.custom_id.split('-')[1]
        }