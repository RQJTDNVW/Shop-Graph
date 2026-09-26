import { motion } from 'framer-motion'
export function AiStatus({state='SHOPGRAPH AI ONLINE'}:{state?:string}){return <motion.div className="ai-status" initial={{opacity:0,y:12}} animate={{opacity:1,y:0}}><span className="pulse"/> {state}</motion.div>}
