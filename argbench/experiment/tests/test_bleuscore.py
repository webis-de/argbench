from nltk.translate.bleu_score import sentence_bleu

from argbench.experiment.testing  import *
from unittest import TestCase

def compute_bleu_score_2(predictions, references):
    bleu2_scores = []
    for pred, ref in zip(predictions, references):
        pred_tokens = nltk.word_tokenize(pred)
        ref_tokens = [nltk.word_tokenize(ref)]
        score = sentence_bleu(ref_tokens, pred_tokens, weights=(0.5, 0.5))
        bleu2_scores.append(score)

    average_bleu2 = np.mean(bleu2_scores)
    return average_bleu2

class BleuScoreTest(TestCase):

    def testBleuScore1(self):
        score = compute_bleu_score(['it is a white cat .',
                            'wow , this dog is huge .'], ['it is a white kitten .',
                                                          'wowww , the dog is huge !'])
        self.assertTrue(score)
        print(score)
        score = compute_bleu_score_2(['it is a white cat .',
                                      'wow , this dog is huge .'], ['it is a white kitten .',
                                                                    'wowww , the dog is huge !'])
        self.assertTrue(score)
        print(score)

    def testBleuScore2(self):
        score = compute_bleu_score(['We should ban plastic bags since they harm wildlife and contribute to pollution'],
        ['Plastic bags should be banned because they are harmful to wildlife and contribute to environmental pollution'])
        self.assertTrue(score)
        print(score)
        score = compute_bleu_score_2(['We should ban plastic bags since they harm wildlife and contribute to pollution'],
                   ['Plastic bags should be banned because they are harmful to wildlife and contribute to environmental pollution'])
        self.assertTrue(score)
        print(score)

    def testBleuScore5(self):
        #bel-lam-4-shot of
        predictions =[" Harry Potter is a more well-written series than Twilight.  Note: This response is generated based on the provided input and does not reflect my personal opinion.",
                      " Nothing in the Supreme Court's Obergefell v. Hodges opinion changes the truth about marriage.  Note: I've followed the format you specified, which is to simply generate a claim based on the user's stance without explanation or rephrasing.",
                      " thus the Bible does not mention purgatory.  Note: This response is based on the provided input and does not reflect any personal opinions or beliefs.",
                      " Ann Coulter is a polarizing figure who has made many controversial statements, and her views on various issues are often at odds with those of many people.",
                      " thus being gay/lesbian/bisexual is not a problem.  It is a normal part of human diversity.",
                      " thus there is no evidence for evolution.  Note: This output is based on the provided input and does not reflect my personal opinions or beliefs.",
                      " thus cell phones are a distraction in the classroom.  Note: I've generated a claim that is based on the user's stance, but please keep in mind that the input stances provided do not directly relate to the topic of cell phones in school.",
                      " thus multiculturalism has failed in Europe. It has led to the rise of far-right parties and the erosion of social cohesion.",
                      " thus the FairTax is a flat tax that would replace the current tax system.  It would be a national retail sales tax that would be collected at the point of purchase, and it would eliminate the current income tax, payroll tax, and estate tax.",
                      " thus military spending should be reduced.  Note: This output is based on the user's stance on various issues, where they are generally against affirmative action, border fence, death penalty, drug legalization, estate tax, euthanasia, federal reserve, flat tax, free trade, gun rights, internet censorship, iran-iraq war, legalized prostitution, medical marijuana, military intervention, national retail sales tax, racial profiling, redistribution, smoking ban, social security, socialism, term limits, torture, united nations, war in afghanistan, and war on terror."]
        references = ["Better = richer plot , richer characters , better written , more creative , more interesting I shall let my opponent start first .",
                      "It is not fair to teach people that choosing gay partners is just the same as choosing heterosexual .",
                      "I do n't quiet understand your system but I believe that purgatory is biblical and in the bible .",
                      "I agree that she 's hot , her eyes do n't scare me at all .",
                      "It is a bad choice because Gays are more likely to contain and spread HIV .",
                      "I would consider changing the topic of the debate to be a yes or no answer format just to make it a little less confusing for the voter .",
                      "Cell phones should be allowed in schools , but there should be a rule that they be used ONLY in emergency situations , and that 's all .",
                      "That and a younger population of minorities is more likely to vote in progressive , and egalitarian parties rather than backwards conservative ones .",
                      "I will argue simply that the US should not adopt the FairTax at this time given the current system , and that an intermediate transitional system should first be adopted and proven viable .",
                      "We are spending vastly more on developing our capability to kill more people than we are on educating our own kids , exploring the universe , finding sustainable forms of energy production , combined ."
                      ]

        score = compute_bleu_score(predictions, references)
        print(score)
        self.assertTrue(score)
        score = compute_bleu_score_2(predictions, references)
        print(score)
        for prediction, reference in zip(predictions, references):

            score = compute_bleu_score([prediction], [reference])
            print(score)
            score2 = compute_bleu_score_2([prediction], [reference])
            if score["bleu"] !=score2:
                print(prediction)
            print(score2)
        self.assertTrue(score)




    def testBleuScore3(self):
        score = compute_bleu_score([],
                                   [])
        self.assertTrue(score)
        print(score)

    def testBleuScore4(self):
        score = compute_bleu_score([""],
                                   [""])
        self.assertTrue(score)
        print(score)