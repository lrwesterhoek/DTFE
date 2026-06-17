
//
// (c) Matthew B. Kennel, Institute for Nonlinear Science, UCSD (2004)
//
// Licensed under the Academic Free License version 1.1 found in file LICENSE
// with additional provisions in that same file.


#include "kdtree2.hpp"

#include <new>
#include <cstdio>
#include <algorithm> 
#include <iostream>

// utility

inline float squared(const float x) {
  return(x*x);
}

inline void swap(int& a, int&b) {
  int tmp;
  tmp = a;
  a = b;
  b = tmp; 
}

inline void swap(float& a, float&b) {
  float tmp;
  tmp = a;
  a = b;
  b = tmp; 
}


//
//       KDTREE2_RESULT implementation
// 

inline bool operator<(const kdtree2_result& e1, const kdtree2_result& e2) {
  return (e1.dis < e2.dis);
}

//
//       KDTREE2_RESULT_VECTOR implementation
// 
float kdtree2_result_vector::max_value() {
  return( (*begin()).dis ); // very first element
}

void kdtree2_result_vector::push_element_and_heapify(kdtree2_result& e) {
  push_back(e); // what a vector does.
  push_heap( begin(), end() ); // and now heapify it, with the new elt.
}
float kdtree2_result_vector::replace_maxpri_elt_return_new_maxpri(kdtree2_result& e) {
  // remove the maximum priority element on the queue and replace it
  // with 'e', and return its priority.
  //
  // here, it means replacing the first element [0] with e, and re heapifying.
  
  pop_heap( begin(), end() );
  pop_back();
  push_back(e); // insert new
  push_heap(begin(), end() );  // and heapify. 
  return( (*this)[0].dis );
}

//
//        KDTREE2 implementation
//

// constructor
kdtree2::kdtree2(kdtree2_array& data_in,bool rearrange_in,int dim_in)
  : the_data(data_in),
    N  ( data_in.shape()[0] ),
    dim( data_in.shape()[1] ),
    sort_results(false),
    rearrange(rearrange_in),
    root(NULL),
    data(NULL),
    ind(N)
{
  if (dim_in > 0)
    dim = dim_in;

  build_tree();

  if (rearrange) {
    // permute the data into index order for cache-friendly searches
    rearranged_data.resize( extents[N][dim] );
    for (int i=0; i<N; i++) {
      for (int j=0; j<dim; j++) {
	rearranged_data[i][j] = the_data[ind[i]][j];
      }
    }
    data = &rearranged_data;
  } else {
    data = &the_data;
  }
}


// destructor
kdtree2::~kdtree2() {
  delete root;
}

// building routines
void kdtree2::build_tree() {
  for (int i=0; i<N; i++) ind[i] = i; 
  root = build_tree_for_range(0,N-1,NULL); 
}

kdtree2_node* kdtree2::build_tree_for_range(int l, int u, kdtree2_node* parent) {
  // recursively build the subtree covering indices [l, u]
  kdtree2_node* node = new kdtree2_node(dim);

  if (u<l) {
    return(NULL); // no data in this node
  }


  if ((u-l) <= bucketsize) {
    // terminal node: compute the true bounding box
    for (int i=0;i<dim;i++) {
      spread_in_coordinate(i,l,u,node->box[i]);
    }

    node->cut_dim = 0;
    node->cut_val = 0.0;
    node->l = l;
    node->u = u;
    node->left = node->right = NULL;


  } else {
    // Approximate bounding box: at the root compute all dimensions; otherwise inherit the parent's
    // box and only recompute the parent's cut dimension.
    int c = -1;
    float maxspread = 0.0;
    int m; 

    for (int i=0;i<dim;i++) {
      if ((parent == NULL) || (parent->cut_dim == i)) {
	spread_in_coordinate(i,l,u,node->box[i]);
      } else {
	node->box[i] = parent->box[i];
      }
      float spread = node->box[i].upper - node->box[i].lower; 
      if (spread>maxspread) {
	maxspread = spread;
	c=i; 
      }
    }

    // c is now the coordinate with the greatest spread; cut on it

    if (false) {
      m = (l+u)/2;
      select_on_coordinate(c,m,l,u);
    } else {
      float sum;
      float average;

      if (true) {
	sum = 0.0;
	for (int k=l; k <= u; k++) {
	  sum += the_data[ind[k]][c];
	}
	average = sum / static_cast<float> (u-l+1);
      } else {
	// alternative: midpoint of the node's box
	average = (node->box[c].upper + node->box[c].lower)*0.5;
      }

      m = select_on_coordinate_value(c,average,l,u);
    }


    node->cut_dim=c;
    node->l = l;
    node->u = u;

    node->left = build_tree_for_range(l,m,node);
    node->right = build_tree_for_range(m+1,u,node);

    if (node->right == NULL) {
      for (int i=0; i<dim; i++) 
	node->box[i] = node->left->box[i]; 
       node->cut_val = node->left->box[c].upper;
       node->cut_val_left = node->cut_val_right = node->cut_val;
    } else if (node->left == NULL) {
      for (int i=0; i<dim; i++) 
	node->box[i] = node->right->box[i]; 
      node->cut_val =  node->right->box[c].upper;
      node->cut_val_left = node->cut_val_right = node->cut_val;
    } else {
      node->cut_val_right = node->right->box[c].lower;
      node->cut_val_left  = node->left->box[c].upper;
      node->cut_val = (node->cut_val_left + node->cut_val_right) / 2.0;
      // recompute the true bounding box as the union of the two subtree boxes
      for (int i=0; i<dim; i++) {
	node->box[i].upper = max(node->left->box[i].upper,
				 node->right->box[i].upper);
	
	node->box[i].lower = min(node->left->box[i].lower,
				 node->right->box[i].lower);
      }
    }
  }
  return(node);
}



void kdtree2:: spread_in_coordinate(int c, int l, int u, interval& interv)
{
  // store the min and max of coordinate c over the indexed data in [l, u] into interv
  float smin, smax;
  float lmin, lmax;
  int i;

  smin = the_data[ind[l]][c];
  smax = smin;


  // process two elements at a time
  for (i=l+2; i<= u; i+=2) {
    lmin = the_data[ind[i-1]] [c];
    lmax = the_data[ind[i]  ] [c];

    if (lmin > lmax) {
      swap(lmin,lmax);
    }

    if (smin > lmin) smin = lmin;
    if (smax <lmax) smax = lmax;
  }
  // handle a trailing odd element
  if (i == u+1) {
    float last = the_data[ind[u]] [c];
    if (smin>last) smin = last;
    if (smax<last) smax = last;
  }
  interv.lower = smin;
  interv.upper = smax;
}


void kdtree2::select_on_coordinate(int c, int k, int l, int u) {
  // partition ind[l..u] so that ind[l..k] are <= ind[k+1..u] along dimension c
  while (l < u) {
    int t = ind[l];
    int m = l;

    for (int i=l+1; i<=u; i++) {
      if ( the_data[ ind[i] ] [c] < the_data[t][c]) {
	m++;
	swap(ind[i],ind[m]); 
      }
    } // for i 
    swap(ind[l],ind[m]);

    if (m <= k) l = m+1;
    if (m >= k) u = m-1;
  }
}

int kdtree2::select_on_coordinate_value(int c, float alpha, int l, int u) {
  // partition ind[l..u] around alpha along dimension c; returns the index of the last element <= alpha
  int lb = l, ub = u;

  while (lb < ub) {
    if (the_data[ind[lb]][c] <= alpha) {
      lb++;
    } else {
      swap(ind[lb],ind[ub]);
      ub--;
    }
  }

  // here ub==lb
  if (the_data[ind[lb]][c] <= alpha)
    return(lb);
  else
    return(lb-1);

}



// search record: per-search scratch state passed down the recursion
static const float infinity = 1.0e38;

class searchrecord {

private:
  friend class kdtree2;
  friend class kdtree2_node;

  vector<float>& qv; 
  int dim;
  bool rearrange;
  unsigned int nn; // , nfound;
  float ballsize;
  int centeridx, correltime;

  kdtree2_result_vector& result;  // results
  const kdtree2_array* data; 
  const vector<int>& ind; 
  // constructor

public:
  searchrecord(vector<float>& qv_in, kdtree2& tree_in,
	       kdtree2_result_vector& result_in) :  
    qv(qv_in),
    result(result_in),
    data(tree_in.data),
    ind(tree_in.ind) 
  {
    dim = tree_in.dim;
    rearrange = tree_in.rearrange;
    ballsize = infinity; 
    nn = 0; 
  };

};


void kdtree2::n_nearest_brute_force(vector<float>& qv, int nn, kdtree2_result_vector& result) {

  result.clear();

  for (int i=0; i<N; i++) {
    float dis = 0.0;
    kdtree2_result e; 
    for (int j=0; j<dim; j++) {
      dis += squared( the_data[i][j] - qv[j]);
    }
    e.dis = dis;
    e.idx = i;
    result.push_back(e);
  }
  sort(result.begin(), result.end() ); 

}


void kdtree2::n_nearest(vector<float>& qv, int nn, kdtree2_result_vector& result) {
  searchrecord sr(qv,*this,result);
  vector<float> vdiff(dim,0.0); 

  result.clear(); 

  sr.centeridx = -1;
  sr.correltime = 0;
  sr.nn = nn; 

  root->search(sr); 

  if (sort_results) sort(result.begin(), result.end());

}


// n nearest neighbors around an existing data point idxin
void kdtree2::n_nearest_around_point(int idxin, int correltime, int nn,
				     kdtree2_result_vector& result) {
  vector<float> qv(dim);  // query vector

  result.clear();

  for (int i=0; i<dim; i++) {
    qv[i] = the_data[idxin][i];
  }

  {
    searchrecord sr(qv, *this, result);
    sr.centeridx = idxin;
    sr.correltime = correltime;
    sr.nn = nn; 
    root->search(sr); 
  }

  if (sort_results) sort(result.begin(), result.end());
    
}


void kdtree2::r_nearest(vector<float>& qv, float r2, kdtree2_result_vector& result) {
// search for all within a ball of a certain radius
  searchrecord sr(qv,*this,result);
  vector<float> vdiff(dim,0.0); 

  result.clear(); 

  sr.centeridx = -1;
  sr.correltime = 0;
  sr.nn = 0; 
  sr.ballsize = r2; 

  root->search(sr); 

  if (sort_results) sort(result.begin(), result.end());
  
}

int kdtree2::r_count(vector<float>& qv, float r2) {
// search for all within a ball of a certain radius
  {
    kdtree2_result_vector result; 
    searchrecord sr(qv,*this,result);

    sr.centeridx = -1;
    sr.correltime = 0;
    sr.nn = 0; 
    sr.ballsize = r2; 
    
    root->search(sr); 
    return(result.size());
  }

  
}

void kdtree2::r_nearest_around_point(int idxin, int correltime, float r2,
				     kdtree2_result_vector& result) {
  vector<float> qv(dim);  // query vector

  result.clear();

  for (int i=0; i<dim; i++) {
    qv[i] = the_data[idxin][i];
  }

  {
    searchrecord sr(qv, *this, result);
    sr.centeridx = idxin;
    sr.correltime = correltime;
    sr.ballsize = r2;
    sr.nn = 0;
    root->search(sr);
  }

  if (sort_results) sort(result.begin(), result.end());

}


int kdtree2::r_count_around_point(int idxin, int correltime, float r2)
{
  vector<float> qv(dim);  // query vector


  for (int i=0; i<dim; i++) {
    qv[i] = the_data[idxin][i];
  }

  {
    kdtree2_result_vector result;
    searchrecord sr(qv, *this, result);
    sr.centeridx = idxin;
    sr.correltime = correltime;
    sr.ballsize = r2;
    sr.nn = 0;
    root->search(sr);
    return(result.size());
  }


}


//
//        KDTREE2_NODE implementation
//

// constructor (the rest of the construction happens during tree building)
kdtree2_node::kdtree2_node(int dim) : box(dim) {
  left = right = NULL;
}

// destructor (box members free themselves)
kdtree2_node::~kdtree2_node() {
  if (left != NULL) delete left;
  if (right != NULL) delete right;
}


void kdtree2_node::search(searchrecord& sr) {
  // Core recursive search: uses true distance to the bounding box to decide whether to descend
  // into the farther child.
  //

  if ( (left == NULL) && (right == NULL)) {
    // terminal node: nn==0 means fixed-radius search, else k-nearest
    if (sr.nn == 0) {
      process_terminal_node_fixedball(sr);
    } else {
      process_terminal_node(sr);
    }
  } else {
    kdtree2_node *ncloser, *nfarther;

    float extra;
    float qval = sr.qv[cut_dim];
    if (qval < cut_val) {
      ncloser = left;
      nfarther = right;
      extra = cut_val_right-qval;
    } else {
      ncloser = right;
      nfarther = left;
      extra = qval-cut_val_left;
    };

    if (ncloser != NULL) ncloser->search(sr);

    // descend into the farther child only if its box can hold a point within ballsize
    if ((nfarther != NULL) && (squared(extra) < sr.ballsize)) {
      if (nfarther->box_in_search_range(sr)) {
	nfarther->search(sr);
      }
    }
  }
}


inline float dis_from_bnd(float x, float amin, float amax) {
  if (x > amax) {
    return(x-amax);
  } else if (x < amin)
    return (amin-x);
  else
    return 0.0;

}

inline bool kdtree2_node::box_in_search_range(searchrecord& sr) {
  // true if any point of this node's bounding box is within sr.ballsize of sr.qv
  int dim = sr.dim;
  float dis2 =0.0; 
  float ballsize = sr.ballsize; 
  for (int i=0; i<dim;i++) {
    dis2 += squared(dis_from_bnd(sr.qv[i],box[i].lower,box[i].upper));
    if (dis2 > ballsize)
      return(false);
  }
  return(true);
}


void kdtree2_node::process_terminal_node(searchrecord& sr) {
  int centeridx  = sr.centeridx;
  int correltime = sr.correltime;
  unsigned int nn = sr.nn; 
  int dim        = sr.dim;
  float ballsize = sr.ballsize;
  //
  bool rearrange = sr.rearrange; 
  const kdtree2_array& data = *sr.data;

  const bool debug = false;

  if (debug) {
    printf("Processing terminal node %d, %d\n",l,u);
    cout << "Query vector = [";
    for (int i=0; i<dim; i++) cout << sr.qv[i] << ','; 
    cout << "]\n";
    cout << "nn = " << nn << '\n';
    check_query_in_bound(sr);
  }

  for (int i=l; i<=u;i++) {
    int indexofi;  // sr.ind[i]; 
    float dis;
    bool early_exit; 

    if (rearrange) {
      early_exit = false;
      dis = 0.0;
      for (int k=0; k<dim; k++) {
	dis += squared(data[i][k] - sr.qv[k]);
	if (dis > ballsize) {
	  early_exit=true; 
	  break;
	}
      }
      if(early_exit) continue; // next iteration of mainloop
      // with rearranged data, accumulate distance before reading the index, so a common early exit
      // (point too far) avoids the index lookup and saves memory bandwidth
      indexofi = sr.ind[i];
    } else {
      // without rearranged data the index is needed up front to address the point
      indexofi = sr.ind[i];
      early_exit = false;
      dis = 0.0;
      for (int k=0; k<dim; k++) {
	dis += squared(data[indexofi][k] - sr.qv[k]);
	if (dis > ballsize) {
	  early_exit= true; 
	  break;
	}
      }
      if(early_exit) continue; // next iteration of mainloop
    } // end if rearrange. 
    
    if (centeridx > 0) {
      // decorrelation interval: skip points too close in index to the center
      if (abs(indexofi-centeridx) < correltime) continue;
    }

    // add the point: push while the result list is undersized, otherwise replace the current max
    if (sr.result.size() < nn) {
      kdtree2_result e;
      e.idx = indexofi;
      e.dis = dis;
      sr.result.push_element_and_heapify(e); 
      if (debug) cout << "unilaterally pushed dis=" << dis;
      if (sr.result.size() == nn) ballsize = sr.result.max_value();  // shrink ball to current largest
      if (debug) {
	cout << " ballsize = " << ballsize << "\n"; 
	cout << "sr.result.size() = "  << sr.result.size() << '\n';
      }
    } else {
      // list is full and this point is closer than the current max, so it replaces it
      kdtree2_result e;
      e.idx = indexofi;
      e.dis = dis;
      ballsize = sr.result.replace_maxpri_elt_return_new_maxpri(e); 
      if (debug) {
	cout << "Replaced maximum dis with dis=" << dis << 
	  " new ballsize =" << ballsize << '\n';
      }
    }
  } // main loop
  sr.ballsize = ballsize;
}

void kdtree2_node::process_terminal_node_fixedball(searchrecord& sr) {
  int centeridx  = sr.centeridx;
  int correltime = sr.correltime;
  int dim        = sr.dim;
  float ballsize = sr.ballsize;
  //
  bool rearrange = sr.rearrange; 
  const kdtree2_array& data = *sr.data;

  for (int i=l; i<=u;i++) {
    int indexofi = sr.ind[i]; 
    float dis;
    bool early_exit; 

    if (rearrange) {
      early_exit = false;
      dis = 0.0;
      for (int k=0; k<dim; k++) {
	dis += squared(data[i][k] - sr.qv[k]);
	if (dis > ballsize) {
	  early_exit=true; 
	  break;
	}
      }
      if(early_exit) continue; // next iteration of mainloop
      // with rearranged data, accumulate distance before reading the index, so a common early exit
      // (point too far) avoids the index lookup and saves memory bandwidth
      indexofi = sr.ind[i];
    } else {
      // without rearranged data the index is needed up front to address the point
      indexofi = sr.ind[i];
      early_exit = false;
      dis = 0.0;
      for (int k=0; k<dim; k++) {
	dis += squared(data[indexofi][k] - sr.qv[k]);
	if (dis > ballsize) {
	  early_exit= true; 
	  break;
	}
      }
      if(early_exit) continue; // next iteration of mainloop
    } // end if rearrange. 
    
    if (centeridx > 0) {
      // decorrelation interval: skip points too close in index to the center
      if (abs(indexofi-centeridx) < correltime) continue;
    }

    {
      kdtree2_result e;
      e.idx = indexofi;
      e.dis = dis;
	sr.result.push_back(e);
    }

  }
}
