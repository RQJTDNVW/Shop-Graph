import type { ApiEnvelope, Product, SearchResult } from '../types'
const base = import.meta.env.DEV
  ? ''
  : (import.meta.env.VITE_API_BASE_URL || '').replace(/\/+$/, '')
async function request<T>(path:string, init?:RequestInit):Promise<T>{
 const res=await fetch(`${base}${path}`,{headers:{'Content-Type':'application/json'},...init}); const body=await res.json().catch(()=>null)
 if(!res.ok) throw new Error(body?.detail || 'The ShopGraph service is unavailable.')
 return body as T
}
export const api={
 health:()=>request<{status:string}>('/api/health'),
 products:(limit=12,offset=0)=>request<ApiEnvelope<{products:Product[];total_products:number}>>(`/api/products?limit=${limit}&offset=${offset}`),
 product:(asin:string)=>request<ApiEnvelope<Product>>(`/api/products/${asin}`),
 search:(query:string)=>request<ApiEnvelope<{query:string;results:SearchResult[];count:number}>>('/api/search',{method:'POST',body:JSON.stringify({query,top_k:24})}),
 recommendations:(user_id='user_001')=>request<ApiEnvelope<unknown>>('/api/recommendations',{method:'POST',body:JSON.stringify({user_id,top_k:12})}),
 profile:(id='user_001')=>request<ApiEnvelope<Record<string,unknown>>>(`/api/users/${id}/profile`),
 history:(id='user_001')=>request<ApiEnvelope<{history:unknown[]}>>(`/api/users/${id}/history`),
 interaction:(event_type:string, product_asin?:string, query?:string)=>request('/api/interactions',{method:'POST',body:JSON.stringify({user_id:'user_001',event_type,product_asin,query})})
}
