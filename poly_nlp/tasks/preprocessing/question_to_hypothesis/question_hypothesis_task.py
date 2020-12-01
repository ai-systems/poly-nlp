from copy import deepcopy
from typing import Dict, List

import stanfordnlp
from conllu import parse
from loguru import logger
from overrides import overrides
from prefect import Task
from tqdm import tqdm

from .rule import AnswerSpan, Question


class QuestionHypothesisExtractionTask(Task):
    def __init__(self, stanford_model_dir, **kwargs):
        super(QuestionHypothesisExtractionTask, self).__init__(**kwargs)
        self.nlp = stanfordnlp.Pipeline(
            processors="tokenize,mwt,pos,depparse",
            models_dir=stanford_model_dir,
        )

    @overrides
    def run(
        self,
        instances,
    ):
        processed_output = {}
        for index, (id, instance) in enumerate(instances.items()):
            try:
                q_doc = self.nlp(instance["question"])
                a_doc = self.nlp(instance["answer"])
                if len(q_doc.sentences) > 1:
                    pre_question = " ".join(
                        word.text
                        for sent in q_doc.sentences[:-1]
                        for word in sent.words
                    )
                else:
                    pre_question = ""
                questions_connlu = parse(
                    "\n".join(
                        f"{index+1}\t{word.text}\t_\t{word.upos}\t{word.xpos}\t_\t{word.governor}\t{word.dependency_relation}\t_\t_"
                        for index, word in enumerate(q_doc.sentences[-1].words)
                    )
                )
                answer_connlu = parse(
                    "\n".join(
                        f"{index+1}\t{word.text}\t_\t{word.upos}\t{word.xpos}\t_\t{word.governor}\t{word.dependency_relation}\t_\t_"
                        for index, word in enumerate(a_doc.sentences[-1].words)
                    )
                )
                question = Question(deepcopy(questions_connlu[0].tokens))
                answer = AnswerSpan(deepcopy(answer_connlu[0].tokens))
                if not question.isvalid:
                    if "___." in instance["question"]:
                        hypothesis = instance["question"].replace(
                            "___.", instance["answer"]
                        )
                    else:
                        hypothesis = instance["question"] + " " + instance["answer"]
                    hypothesis = f"{pre_question} {hypothesis}".lstrip()
                    # logger.warning(f"Replacing with {hypothesis}")
                    processed_output[id] = hypothesis
                    continue
                if not answer.isvalid:
                    logger.error(f"Answer: {instance['answer']} is invalid")
                    continue
                # print(instance["question"])
                # print(instance["answer"])
                question.insert_answer_default(answer)
                hypothesis = " ".join(question.format_declr())
                hypothesis = f"{pre_question} {hypothesis}".lstrip()
                processed_output[id] = hypothesis
            except AssertionError:
                logger.error(f"Question not available {instance['question']}")
                processed_output[id] = hypothesis
        return processed_output
