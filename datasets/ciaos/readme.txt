rating_with_timestamp.mat

rating.mat includes the rating information with time points when ratings are created. there are six columns and they are userid, productid, categoryid, rating, helpfulness and  time point, respectively. 

*****************************************************************
For example, for one row
(1,2,3,4,5, 6)

It means that user 1 gives a rating of 4 to the product 2 from the category 3. The helpfulness of this rating is 5. 6 is the number of seconds from 1/1/1970 0-0-0 to the time point the rating created.

*****************************************************************************


===============================================================================================
trustnetwork.mat

trustnetwork.mat includes the trust relations between users. There are two columns and both of them are userid.

*************************************************************************************
for example, for one row,
(1,2)

it means that user 1 trusts user 2.
*************************************************************************************



----------------------------------------------------------------------------------------------------------------
References
----------------------------------------------------------------------------------------------------------------


@Conference{tang-etal12a,
  title={m{T}rust: {D}iscerning multi-faceted trust in a connected world},
  author={Tang, J. and Gao, H. and Liu, H.},
  booktitle={Proceedings of the fifth ACM international conference on Web search and data mining},
  pages={93--102},
  year={2012},
  organization={ACM}
}


@Confernce{tang-etal12b,
  title={e{T}rust: {U}nderstanding trust evolution in an online world},
  author={Tang, J.and Gao, H. and Liu, H.  and Das Sarma, A.},
  booktitle={Proceedings of the 18th ACM SIGKDD international conference on Knowledge discovery and data mining},
  pages={253--261},
  year={2012},
  organization={ACM}
}