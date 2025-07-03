from langchain.prompts import (
    PromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
)

SYSTEM_TEMPLATE_DEFAULT = (
    "You are a very precise, and knowledgeable statistics researcher."
    "Your job is to use summarize reports."
    "Be as precise and truthful to the reports as possible." 
    "Format your response in the following structure:\n" 
    "SUMMARY:\n<comprehensive summary here>\n\n"
    "KEYWORDS:\n<comma-separated keywords>\n\n"
    "TAGS:\n<comma-separated tags>"
    "Reports: {reports}"
)

HUMAN_TEMPLATE_DEFAULT = (
    "Please analyze the reports and provide:\n"
    "1. A comprehensive summary\n"
    "2. {keywords} most important keywords\n"
    "3. {tags} relevant tags\n\n"
    "4. With a maximum of {max_words} words"
    "5. Provide the output in the following language: {out_lang}"
)


class LangChainPromptTemplate:

    def __init__(self, 
                 system_template: str = SYSTEM_TEMPLATE_DEFAULT,
                 human_template:str =  HUMAN_TEMPLATE_DEFAULT
                 ) -> None:
        # Creating a summary Template
        self.system_template = system_template
        self.system_prompt = SystemMessagePromptTemplate(
            prompt=PromptTemplate(input_variables=["reports"], template=self.system_template)
        )
        self.human_prompt = HumanMessagePromptTemplate(
                prompt=PromptTemplate(
                    input_variables=["keywords","tags","max_words","out_lang"], 
                    template = human_template
                )
        )

    def create_prompt(self, human:bool = True):
        if human: 
            # for the task of summarising, I think the human prompt could be optional.
            messages = [self.system_prompt, self.human_prompt]
            return ChatPromptTemplate(
                input_variables=["reports", "keywords","tags","max_words","out_lang"],
                messages=messages,
            )
        else:
            messages = [self.system_prompt]
            return ChatPromptTemplate(
                input_variables=["reports"],
                messages=messages,
            )

if __name__ == "__main__":
    template_maker = LangChainPromptTemplate()
    reports = """O Índice de Produção na Construção1 aumentou 1,9 em abril, variação superior em 0,7 pontos percentuais
    (p.p.) à observada no mês anterior.
    O índice de emprego acelerou 0,3 p.p., para 3,0%, e as remunerações aumentaram 2,1 p.p., para uma variação
    homóloga de 10,0%"""
    print(template_maker.create_prompt().format_messages(reports=reports, 
                                                   keywords = 5,tags=1,max_words=4,out_lang="pt-pt"))
