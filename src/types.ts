export type Product = { parent_asin?:string; asin?:string; title?:string; main_category?:string; store?:string; price?:number|string; average_rating?:number; rating_number?:number; images?:unknown; image_url?:string; imageURLHighRes?:string[]; details?:Record<string,string>; description?:string|string[]; [key:string]: unknown }
export type SearchResult = Product & { score?:number; similarity?:number; relevance_score?:number }
export type ApiEnvelope<T> = { success:boolean; data:T; processing_time_ms?:number }
