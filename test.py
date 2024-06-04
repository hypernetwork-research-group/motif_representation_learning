from datasets import EmailEnron, EmailEnronFull, EmailEuFull, ContactHighSchool, ContactPrimarySchool, NDCClassesFull, TagsAskUbuntu, TagsMathSx
dataset = EmailEnronFull()
incidence_matrix = dataset.incidence_matrix(lambda e: len(e) > 1)
print(dataset)

from pymochy import Mochy
from motif import motif_negative_sampling

mochy = Mochy(incidence_matrix)

motifs = mochy.sample(2)[:, 1:]
