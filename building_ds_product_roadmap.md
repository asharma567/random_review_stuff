
What's the best way to create/augment a product roadmap?
------------------

vertical innovation: (v1-vn) creating a faster horse 
lateral innovation: (v0) creating a car

data science tools can be used for either topline/bottomline augmentation. Topline finding products that bring in revenue like new customer acquisition (marketing).


Empathize w the end user! (important to do before building anything)
- using the actual product with as many scenarios as possible.

- CS, about complaints/reviews. This channel should surface all the frictional points of the product. Be careful ot be data driven as anecdotal feedback might be subject to bias like recency, data-driven approach Using a topic modeler on the corpus of reviews for the past X months

- speaking with product people to see what they've already researched. Inpect preexisting product roadmap and see where data science can be inserted instead of analytical/engineering tools. Looking for v1-vn improvements. Gather data from product analytics. (this is better for vertical innovation)

...from both these channel's I'd consider vertical/lateral growth.

How to best deal with prioritization?
--------------------------
- The most ideal way is to create a cost-benefit structure to each task. 

Costs, being the time it takes for the average IC to do the task.

Benefits, are much more difficult to gauge. It requires knowledge of it's impact on the revenue stream.

It might be worth figuring out business metrics of success for this project early-on. Obviously something far along the funnel such as revenue isn't direct enough, i.e. it's too noisy. So pick something that's 1:1 impactful causal to the product. 

Measuring & communicating success of the product is an entirely different subject. There are rudamentry ways of computing like a/b tests on metric of choice, e.g. "lift".

To aid in prioritization we constructed something called "impact chart."

```
project categories
----
A) Budgets 
B) Customer Success 
C) Internal & Scalability 
D) Marketing & Sales 
E) R & D

Lead Scoring: D Client Churn modeling: B

- Leading indicators 
  - Consider using NPS data as a data source
- Survey design
Data capture: C

- Pre-Existing products explore which metrics needs to be captured by engineering
- Explore viable health metrics for pre-existing DATA PRODUCTs
ROI calculator: BD

- Explanation of what rocketrip as a product means to a client from an ROI perspective. Currently KB and Forest produce a report to aid sales with thier pitch.
Subsequent iterations on pre-existing data products

- Flight budget: A
    Taking more dimensions into account versus just pricing data

- Hotel budget: A
    More intelligent determination of which geographic regions are appropriate for hotel budgeting
    More intelligent determination of which hotel brands are appropriate for hotel budgeting
    Improve star class mapping between Northstar and Expedia/Priceline data sets
    Infer intentions from entered location
      - If it's a city center, budget where appropriate generally in that city
      - If it's a specific location, determine an appropriate area to find a hotel
(Approvals) Fully automonous points rewarding: C

This is one of the finalizing components to automating the Approval's process. It's a model designed to interpret and validate the saver scenario at hand and undlying expenses behavior.
Fraud gaming: BCD

Piggy-backs on the points rewards phase of the Approval's model. An anomaly detection system could automated, which w...

```



How do you gauge the difficulty of a project/task?
--------------------------------------------
Predict the number of hours it would take a new grad to do this work:

Set of power point slides or computing a report (8 hours or so).

writing paper or blog on a dataproduct, a much longer time. 

it requires domain expertise, experience in writing analytically/qualitatively, knowledge of algorithm and it's application. 

How data science could impact topline growth?
-----------------------
Mitigating retention/churn and plugging up major holes in the balance sheet is one thing, e.g. if one has to off shore a bunch of work.

Marketing/new client acquisition is another way. that is identifying the cohort of clients to go after. This could work at either a b2c, c2c. Customer segmentation is agnostic to distribution. 

However, the distribution tools would be different:
- Marketo/Sendgrid for emails which could be automated
- sales org can be supported by client targets and analytical presentations.

