import { Heart, Plus, Sparkles, Star } from 'lucide-react'
import { motion } from 'framer-motion'
import type { Product } from '../types'
const firstUrl=(value:unknown):string=>{
 if(Array.isArray(value)) return firstUrl(value[0])
 if(typeof value!=='string') return ''
 const match=value.match(/https?:\/\/[^\s'"\]]+/)
 return match?.[0]||''
}
const imageOf=(p:Product)=> firstUrl(
 p.image_url || p['images.hi_res'] || p['images.large'] || p['images.thumb'] ||
 (p.imageURLHighRes as string[]|undefined)?.[0] ||
 (Array.isArray(p.images) ? (p.images[0] as {large?:string})?.large : undefined)
)
const priceOf=(p:Product)=>{const v=typeof p.price==='number'?p.price:parseFloat(String(p.price||'')); return Number.isFinite(v)?`$${v.toFixed(2)}`:'View price'}
export function ProductCard({product,onCart,onOpen,index=0}:{product:Product;onCart:(p:Product)=>void;onOpen:(p:Product)=>void;index?:number}){
 const score=Number(product.score ?? product.similarity ?? product.relevance_score); const pct=Number.isFinite(score)?Math.min(99,Math.max(76,Math.round(score<=1?score*100:score))):null
 const image=imageOf(product)
 return <motion.article className="product-card" initial={{opacity:0,y:16}} animate={{opacity:1,y:0}} transition={{delay:index*.045}} whileHover={{y:-5}}>
   <button className={`image-wrap${image?'':' no-image'}`} onClick={()=>onOpen(product)} aria-label={`View ${product.title||'product'}`}>{image&&<img src={image} loading="lazy" onError={e=>{(e.currentTarget.parentElement!).classList.add('no-image')}}/>}<span className="placeholder">{(product.title||'SG').slice(0,2)}</span></button>
   <div className="product-copy"><div className="eyebrow">{String(product.store||product.main_category||'ELECTRONICS')}</div><h3 onClick={()=>onOpen(product)}>{String(product.title||'Untitled product')}</h3><div className="rating"><Star size={14} fill="currentColor"/>{Number(product.average_rating||0).toFixed(1)} <span>({Number(product.rating_number||0).toLocaleString()})</span></div>
   {pct&&<div className="match"><Sparkles size={13}/><span>AI match {pct}%</span><i><b style={{width:`${pct}%`}}/></i></div>}
   <div className="card-footer"><strong>{priceOf(product)}</strong><div><button className="icon-btn" aria-label="Save product"><Heart size={17}/></button><button className="add-btn" onClick={()=>onCart(product)} aria-label="Add to cart"><Plus size={18}/></button></div></div></div>
 </motion.article>
}
