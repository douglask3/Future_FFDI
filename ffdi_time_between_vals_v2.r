graphics.off()
country = 'Australia'
file = paste0("outputs/cal_time-", country, ".csv")
dat = read.csv(file)
 
 rcps = c("2.6" = "rcp2_6", "8.5" = "rcp8_5")
 degs = c("Baseline" = -1, "None" = 0, "1_5" = 1.5, "2_deg" = 2, "4_deg" = 4)
 cols = rev(c('#ffffb2','#fecc5c','#fd8d3c','#f03b20','#bd0026'))
 diff = FALSE
 xlog = TRUE
 tmin = 1/24
 
 cal_plot_data <- function(rcp, deg) {
	index = which(grepl(rcp, colnames(dat)) & grepl(deg, colnames(dat)))
	index0 = which(grepl(rcp, colnames(dat)) & grepl("Baseline", colnames(dat)))
	out = dat[index] 
	
	if (deg != 'Baseline') out = out + dat[index0]
	#else {
		out = out[nrow(out):1,]
		#out = t(sapply(2:nrow(out), function(i) out[i,] + apply(out[1:(i-1),], 2, sum)))
		#out = apply(out, 1, function(i) 365*unlist(i)/unlist(out[nrow(out),]))
		#out = t(out)
		#out = out
	#}	
	out = as.matrix(out)
	
	return(out)
}

dat = lapply(names(degs), function(deg) lapply(rcps, cal_plot_data, deg))

plot_rcpDeg <- function(rcp_i, deg_j) {
        
	if (rcp_i == 1 && deg_j == 1) return()
#	if (rcp_i == 2 && deg_j == 2) return()
	y = (deg_j-1) * (2.5+length(rcps)) + 2*rcp_i - 3
	yl = rep(y, 2)
	tdat = dat[[deg_j]][[rcp_i]]
	tdat = rbind(0, tdat)
	if (xlog) pdat = log10(tdat + tmin)
	else pdat = tdat
        
        if (deg_j > 1) tdat = tdat - rbind(0, dat[[1]][[rcp_i]])
	plot_lines <- function(i) {
		plot_line <- function(j) {
			lines(c(pdat[i-1, j], pdat[i, j]), yl -0.5 + j/ncol(pdat), 
				  lwd = 5, col = cols[i-1])
			
		}
		if (i > 1) lapply(1:ncol(pdat), plot_line)
		#if (i == nrow(pdat)) return()
		if (deg_j > 1) 
		    txt = round(tdat[i,], 2) #  - dat[[1]][[1]][i-1,]
		else 
		    txt = round(tdat[i,], 2)
			
		txt = sort(txt)
		txt = paste0(txt[2], '\n(', txt[1], '-', txt[3], ')')
                
		text(pdat[i,3], yl + 1.2, txt, cex = 0.6, xpd = TRUE)
	}
	lapply(2:nrow(pdat), plot_lines)
}

png(paste0("time_in_ffdi_range", country, ".png"), height = 5, width = 7.2, units = 'in', res = 300)
par(mar = c(2, 3, 0.1, 2.2), oma = c(0, 1, 0, 0))
xrange = c(0, max(unlist(dat, recursive = TRUE), na.rm = TRUE))
if (xlog) xrange = log10(xrange + tmin)
plot(xrange, c(0, length(degs) * (2.5+length(rcps))), axes = FALSE,
	 type = 'n', xlab = '', ylab = '', xaxs = 'i')
	  
lapply(1:length(degs), function(j) lapply(1:length(rcps), plot_rcpDeg, j))
						
if (xlog) {
	at = c(0, 1/24, 1/4, 1, 7, 30, 365)
	at = log10(at + tmin)
	labels = c(0, 'hour', '6-hr', 'day', 'week', '~month', 'year')
	axis(1, at = at, labels = labels)
} else axis(1)

text(x = xrange[1], y = 0.5, '1986-\n2005', 
	 srt = 90, xpd = NA, adj = c(0.5, -1))
text(x = xrange[1], y = 4.5, '2004-\n2023', 
	 srt = 90, xpd = NA, adj = c(0.5, -1))
text(x = xrange[1], 
	 y =  4.5+ (length(rcps) + 2.5) * (1:(length(degs)-2)), 
	 degs[-(1:2)], 
	 srt = 90, xpd = TRUE, adj = c(0.5, -2.75))
for (i in 1:length(rcps)) {
	text(x = xrange[1], 
		 y = i*2 - 3 + (length(rcps) + 2.5) * (1:(length(degs)-1)), 
	     names(rcps)[i], 
	 srt = 90, xpd = TRUE, adj = c(0.5, -1.25))
}
#axis(2, at = 0.5+ (length(rcps) + 1) * (1:(length(degs)-1)), labels = degs[-1])


dev.off()
