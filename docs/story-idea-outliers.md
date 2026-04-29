# Story idea: The flats that defy the algorithm

## The pitch

A regression model that controls for town, flat type, size, floor, remaining lease, distance to MRT/CBD/schools/hawker centres, and even superstition effects can predict 93.7% of HDB resale prices within ~$27,000. But for some flats, it's off by $300,000–$450,000. These are the transactions no amount of data can explain — the ones where you need to knock on the door and ask.

The outliers are the story. The model is the reporting tool.

## The alamak flats (sold way above prediction)

These flats sold for 30-45% more than what every measurable factor says they should be worth. The model sees "Bukit Merah, 5-room, floor 25, 89 years lease" and predicts $1.2M. It actually sold for $1.65M. Why?

**Patterns from Model 12's top outliers:**

### The Boon Tiong Road cluster
- Blk 9A and 9B Boon Tiong Road dominate the top 20 — four entries from the same two blocks
- These are Improved 5-room flats in Bukit Merah near SkyTerrace@Dawson
- Selling for $1.45M–$1.65M when the model predicts $1.1M–$1.2M
- **Possible drivers:** Redevelopment speculation (SERS?), unblocked views, proximity to the Dawson rejuvenation, or exceptional renovation. Worth checking if there's been recent BTO/SERS announcements nearby.

### The heritage premium
- Blk 5 Changi Village Road: 3-room flat at $495K, model predicts $96K (+416%). The single biggest percentage outlier. Changi Village is a unique heritage enclave — seafood, beach, kampung vibe — that the "Pasir Ris" town dummy can't capture.
- This is effectively a lifestyle premium the data has no variable for.

### The view premium
- Blk 241 Bishan St 22 (Executive, $1.45M, predicted $999K, +45%): near Bishan-AMK Park with likely unblocked views
- Marine Parade blocks: sea-facing units where `coast_dist_m` captures proximity but not "actual sea view from the living room"

### The DBSS/micro-neighbourhood effect
- Pine Close cluster (Blks 1, 3, 5, 7): four separate 5-room flats, all 30-38% above prediction. These are DBSS flats near Braddell MRT. The consistency across four transactions suggests a genuine micro-neighbourhood premium, not random noise.
- Clementi Ave 3 (Blks 445A, 445B): two transactions, both 33-38% above. Same pattern — a specific cluster of blocks commanding premiums the town-level dummy can't see.

## The bargain flats (sold way below prediction)

These sold for 25-38% less than expected. Possible distress sales, estate sales, condition issues, or structural problems.

**Patterns from Model 12's bottom outliers:**

### The Jalan Ma'mor terrace problem
- Blk 53 Jalan Ma'mor: 367 sqm terrace at $1.57M, predicted $2.3M (-32%). This is the model's Achilles heel — a landed house classified as a 3-room HDB flat. Even with the terrace interaction in the log model, 367 sqm is so far outside the training range that extrapolation fails.

### The 3Gen flat stigma
- Boon Lay Ave (Blks 216A, 217A, 218A, 218D): five entries, all 3Gen 5-room flats selling for $640K-$725K when predicted at $930K-$1M
- Buangkok Crescent (Blks 997A, 997B, 997C): five entries, same pattern
- The 3Gen design (extra granny room, awkward layout) carries a $250K-$300K market discount that the flat_model dummy can't fully absorb. Agents report these take longer to sell and require significant price cuts.
- **Reporting angle:** Why does the market reject 3Gen flats so heavily? Interview buyers who walked away and sellers who had to cut.

### Possible distress/estate sales
- Blk 726 Tampines St 71: 5-room at $518K, predicted $833K (-38%). An ordinary location with an extraordinary discount. No structural reason for this — could be a distress sale, estate sale, or unit in very poor condition.
- Blk 8 Empress Rd (Bukit Timah): 3-room at $438K, predicted $716K (-39%). Bukit Timah is the most expensive town in the model. A flat selling at 61% of predicted price here is highly unusual.
- **Reporting angle:** These are real people selling at huge discounts for unknown reasons. Worth investigating — are these divorces, estate settlements, ethnic integration policy constraints?

## The reporting framework

1. **The model surfaces the needles.** Out of 50,718 transactions, Model 12 flags ~100 that deviate by more than 25%. These are the starting points.
2. **The reporter explains the hay.** For each flagged transaction, the question is: what does this flat have (or lack) that no dataset captures? Renovation? Views? Heritage charm? Divorce? SERS speculation?
3. **The story is the gap.** "A model that accounts for every measurable factor — location, size, floor, lease, proximity to MRT and schools — still can't explain why this flat sold for $450,000 more than its neighbours. We went to find out."

## Possible formats

- **Individual flat profiles:** Pick the 3-5 most interesting outliers, visit them, photograph them, talk to agents/neighbours. "The $1.65M flat in Boon Tiong Road that no algorithm saw coming."
- **The 3Gen investigation:** A focused piece on why the market rejects 3Gen flats. 10+ entries in the bargain list. Policy implications — HDB designed these for multi-generational living, but buyers don't want them.
- **The heritage tax:** Changi Village, Tiong Bahru walk-ups, Stirling Road terraces — flats where the "HDB" label massively undersells what they actually are. The model predicts HDB prices; these are heritage homes.
- **Interactive:** Publish the model's predictions alongside actual prices. Let readers look up any recent HDB transaction and see whether their flat was an alamak or a bargain. (Privacy considerations: this is public data from data.gov.sg, but framing matters.)

## What you'd need

- Visit the top alamak and bargain flats (or at least their blocks)
- Talk to property agents who handled transactions in those blocks
- Check for SERS/BTO announcements near the alamak clusters
- Check court records for the biggest bargain flats (estate/divorce sales?)
- Photographs of the blocks and surroundings
