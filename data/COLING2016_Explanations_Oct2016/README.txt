=============================================================================================================
                                                README
=============================================================================================================

This archive contains knowledge resources and data for the paper: 

"What's in an Explanation? Characterizing Knowledge and Inference Requirements for Elementary Science Exams"
by Peter Jansen, Niranjan Balasubramanian, Mihai Surdeanu, and Peter Clark (COLING 2016).




=============================================================================================================
                                            KNOWLEDGE RESOURCES
=============================================================================================================

* Gold Explanations, and AKBC'13 knowldege-type annotation:
  Elementary-NDMC-Train-WithExplanations.csv
  
* Relation annotation for explanation sentences:
  relationannotation/annotationOct2016.txt
  
  This file contains each sentence listed in a gold explanation, but only those within the following file
  contain annotation (approximately 50% of original questions):
  Elementary-NDMC-Train-WithExplanations-SubsetWithRelationAnnotation.csv
  
  
=============================================================================================================
                                        GUIDES AND DOCUMENTATION
=============================================================================================================
  
Plain-language guides and documentation for the explanation generation process, relation annotation, as well
as replicating the AKBC2013 knowledge type annotation on this larger set of questions, can be found in the
following documents:

    annotationguides/explanation_relation_annotation_guide.pdf
    knowledge_type_annotation_guide.pdf
    

    
=============================================================================================================
                                                TOOLS
=============================================================================================================

The graphical annotation tool used to perform the relation annotation on the explanation sentences in this
archive is included in the following location:

    annotationtool/annotationtool.py
    
It was developed using Python 3.5.1 and the tkInter GUI library version 8.6 .  



=============================================================================================================
                                        COMMENTS, QUESTIONS, ISSUES
=============================================================================================================

Please contact the corresponding author, Peter Jansen (pajansen@email.arizona.edu)

