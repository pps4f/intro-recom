package org.grouplens.mooc.cbf;

import com.google.common.collect.Maps;
import it.unimi.dsi.fastutil.longs.LongSet;
import it.unimi.dsi.fastutil.longs.LongSortedSet;
import org.grouplens.lenskit.core.Transient;
import org.grouplens.lenskit.vectors.MutableSparseVector;
import org.grouplens.lenskit.vectors.SparseVector;
import org.grouplens.lenskit.vectors.VectorEntry;
import org.grouplens.mooc.cbf.dao.ItemTagDAO;

import javax.inject.Inject;
import javax.inject.Provider;
import java.util.Map;
import java.util.Set;
import java.util.List;

/**
 * Builder for computing {@linkplain TFIDFModel TF-IDF models} from item tag data.  Each item is
 * represented by a normalized TF-IDF vector.
 *
 * @author <a href="http://www.grouplens.org">GroupLens Research</a>
 */
public class TFIDFModelBuilder implements Provider<TFIDFModel> {
    private final ItemTagDAO dao;

    /**
     * Construct a model builder.  The {@link Inject} annotation on this constructor tells LensKit
     * that it can be used to build the model builder.
     *
     * @param dao The item-tag DAO.  This is where the builder will get access to items and their
     *            tags.
     *            <p>{@link Transient} means that the provider promises that the DAO is no longer
     *            needed once the object is built (that is, the model will not contain a reference
     *            to the DAO).  This allows LensKit to configure your recommender components
     *            properly.  It's up to you to keep this promise.</p>
     */
    @Inject
    public TFIDFModelBuilder(@Transient ItemTagDAO dao) {
        this.dao = dao;
    }

    /**
     * This method is where the model should actually be computed.
     * @return The TF-IDF model (a model of item tag vectors).
     */
    @Override
    public TFIDFModel get() {
        // Build a map of tags to numeric IDs.  This lets you convert tags (which are strings)
        // into long IDs that you can use as keys in a tag vector.
        Map<String, Long> tagIds = buildTagIdMap();
        System.out.println("tagIds.size() = " + tagIds.size());

        /*for(Map.Entry<String, Long> entry: tagIds.entrySet()) {
            System.out.println(entry.getKey());
        } */

        // Create a vector to accumulate document frequencies for the IDF computation
        MutableSparseVector docFreq = MutableSparseVector.create(tagIds.values());
        docFreq.fill(0);

        // We now proceed in 2 stages. First, we build a TF vector for each item.
        // While we do this, we also build the DF vector.
        // We will then apply the IDF to each TF vector and normalize it to a unit vector.

        // Create a map to store the item TF vectors.
        Map<Long,MutableSparseVector> itemVectors = Maps.newHashMap();
        System.out.println("itemVectors.size() = " + itemVectors.size());

        // Create a work vector to accumulate each item's tag vector.
        // This vector will be re-used for each item.
        MutableSparseVector work = MutableSparseVector.create(tagIds.values());
        System.out.println("884: " + tagIds.get("3"));
        LongSortedSet keys = work.keyDomain();
        System.out.println("key.size(): " + keys.size());
        //System.out.println("containskey " + work.containsKey(884));
        // Iterate over the items to compute each item's vector.
        LongSet items = dao.getItemIds();
        for (long item: items) {
            // Reset the work vector for this item's tags.
            work.clear();
            // Now the vector is empty (all keys are 'unset').

            // TODO Populate the work vector with the number of times each tag is applied to this item.
            // TODO Increment the document frequency vector once for each unique tag on the item.
            List<String> itemtags = dao.getItemTags(item);
            for(String tag : itemtags) {
                long tagid = tagIds.get(tag);
                //System.out.println("tag: " + tag + "; tagid: " + tagid);
                //double tagcount = work.get(tagid);
                //System.out.println("containskey " + work.containsKey(tagid));
                if (!work.containsKey(tagid)) {
                    work.set(tagid, 1.0);
                    docFreq.add(tagid, 1.0);
                }
                else {
                    work.add(tagid, 1.0);
                }
            }

            System.out.println("item: " + item + "; work.size(): " + work.size());

            //if (work.containsKey(tagIds.get("Tom Hanks"))) {
            //    System.out.println("item: " + item + "; Tom Hanks = " + work.get(tagIds.get("Tom Hanks")));
            //}

            // Save a shrunk copy of the vector (only storing tags that apply to this item) in
            // our map, we'll add IDF and normalize later.
            itemVectors.put(item, work.shrinkDomain());
            // work is ready to be reset and re-used for the next item
        }
        System.out.println("Number of items = " + items.size());

        // Now we've seen all the items, so we have each item's TF vector and a global vector
        // of document frequencies.
        // Invert and log the document frequency.  We can do this in-place.
        for (VectorEntry e: docFreq.fast()) {
            // TODO Update this document frequency entry to be a log-IDF value
            long key = e.getKey();
            double val = e.getValue();
            double newval = Math.log(items.size()/val);
            docFreq.set(key, newval);
            //System.out.println("key: " + key + "; val: " + val + "; newval:" + newval);
        }

        // Now docFreq is a log-IDF vector.
        // So we can use it to apply IDF to each item vector to put it in the final model.
        // Create a map to store the final model data.

        Map<Long,SparseVector> modelData = Maps.newHashMap();
        for (Map.Entry<Long,MutableSparseVector> entry: itemVectors.entrySet()) {

            MutableSparseVector tv = entry.getValue();
            double vectlen = tv.norm();
            System.out.println(vectlen);
            // TODO Convert this vector to a TF-IDF vector
            // TODO Normalize the TF-IDF vector to be a unit vector
            // HINT The method tv.norm() will give you the Euclidian length of the vector
            for (VectorEntry e: tv.fast()) {
                long key = e.getKey();
                double val = e.getValue();
                double newval = (val / docFreq.get(key))/vectlen;
                System.out.println(
                        "key: " + key +
                        "; val: " + val +
                        "; IDF: " + docFreq.get(key) +
                        "; vectlen: " + vectlen +
                        "; newval:" + newval);
                tv.set(key, newval);
            }

            // Store a frozen (immutable) version of the vector in the model data.
            modelData.put(entry.getKey(), tv.freeze());
        }

        //System.out.println("i: " + i + "; j: " + j);

        // we technically don't need the IDF vector anymore, so long as we have no new tags
        return new TFIDFModel(tagIds, modelData);
    }

    /**
     * Build a mapping of tags to numeric IDs.
     *
     * @return A mapping from tags to IDs.
     */
    private Map<String,Long> buildTagIdMap() {
        // Get the universe of all tags
        Set<String> tags = dao.getTagVocabulary();
        // Allocate our new tag map
        Map<String,Long> tagIds = Maps.newHashMap();

        for (String tag: tags) {
            // Map each tag to a new number.
            tagIds.put(tag, tagIds.size() + 1L);
        }
        return tagIds;
    }
}
